from models import *
from data import *
from tqdm import tqdm
from trainer import Trainer
from evaluator import Evaluator
from lime.lime_tabular import LimeTabularExplainer
import tensorflow_addons as tfa


class Experiment:
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.stl_epochs = args.stl_epochs
        self.mtl_epochs = args.mtl_epochs
        self.regression = args.regression
        self.es_patience = args.es_patience
        self.pl_patience = args.pl_patience
        self.runs = args.runs
        self.verbose = args.verbose
        self.tune_arch = args.tune_arch
        self.show_full_scores = args.show_full_scores
        self.lime_instance = None
        self.save_plots = None

        self.stl_model_trainer, self.mtl_model_trainer = None, None
        self.stl_model_evaluator, self.mtl_model_evaluator = None, None

    def plot(self, full_runs, stl_score, stl_fid, scores_list, fid_list):
        import matplotlib.pyplot as plt
        import pickle
        from paretoset import paretoset

        pareto_input = full_runs[['Mean Fidelity',
                                  f'Mean {"MSE" if self.regression else "Accuracy"}']].values.astype(float)
        pareto_mask = paretoset(pareto_input, sense=['min', f'{"min" if self.regression else "max"}']).flatten()
        pareto_mask_w_stl = np.insert(pareto_mask, 0, False)

        xs = [stl_score] + scores_list
        ys = [stl_fid] + fid_list

        max_score = max(xs)
        max_fid = max(ys)

        colors = pickle.load(open('fig_colors.pkl', 'rb'))

        labels = ['STL', 'α = 0.1', 'α = 0.2', 'α = 0.3', 'α = 0.4',
                  'α = 0.5', 'α = 0.6', 'α = 0.7', 'α = 0.8', 'α = 0.9']

        markers = ['X'] + ['o' for _ in range(len(labels) - 1)]
        fig = plt.figure(figsize=(8, 6), )
        plt.plot(xs[1:], ys[1:], '--', zorder=1, alpha=0.4)  # do not include STL
        for i in range(len(labels)):
            s = 100 if i == 0 or pareto_mask_w_stl[i] else None
            e = 'black' if pareto_mask_w_stl[i] else 'face'
            plt.scatter(xs[i], ys[i], c=np.array(colors[i]).reshape(1, -1), label=labels[i], zorder=2,
                        marker=markers[i], s=s, edgecolors=e)

        upper_limit = 0.01 if self.regression else 0.1
        lower_limit = 0.01
        plt.xlim(stl_score - abs(stl_score - (max_score + lower_limit)),
                 stl_score + abs(stl_score - (max_score + upper_limit)))
        plt.ylim(0,
                 stl_fid + abs(stl_fid - (max_fid + 0.01)))
        plt.xlabel(f'{"MSE" if self.regression else "Accuracy"}')
        plt.ylabel('Global Fidelity')
        plt.grid()
        plt.legend()
        plt.title(self.dataset_name)

        fig.savefig(self.dataset_name + '_plot.pdf', bbox_inches='tight')

    def run(self):
        # run linear baseline
        self.run_linear()
        print('\n*************************************************************************************')
        stl_model, stl_score, stl_fid, stl_surrogate = self.run_stl()
        # print('\n*************************************************************************************')
        # self.run_mtl(stl_model, stl_score, stl_fid)
        print('\n*************************************************************************************')
        self.run_decoupled(stl_model)
        # print('\n*************************************************************************************')
        # self.run_decoupled_mtl(stl_model)
        print('\n*************************************************************************************')
        self.run_regularized(stl_model)

    def run_linear(self):
        print('\nLinear baseline training...')
        x_train, y_train, x_test, y_test, _, _, _ = get_data(
            self.dataset_name
        )

        linear_bsl_trainer = Trainer(
            model=linear_baseline(self.regression), surrogate=None, x_train=x_train, y_train=y_train,
            regression=self.regression
        )

        linear_bsl = linear_bsl_trainer.train(epochs=self.stl_epochs, tune=False,
                                              verbose=self.verbose)
        linear_bsl_evaluator = Evaluator(black_box=linear_bsl, surrogate=None,
                                         x_test=x_test, y_test=y_test)
        linear_bsl_evaluation = linear_bsl_evaluator.mse() if self.regression else linear_bsl_evaluator.accuracy()
        print(f'\nTest {"MSE" if self.regression else "Accuracy"} score of linear baseline:',
              linear_bsl_evaluation)

    def run_stl(self):
        stl_test_fid = list()
        stl_test_r2_models = list()

        stl_test_scores1, stl_test_scores2 = list(), list()

        x_train, y_train, x_test, y_test, _, _, _ = get_data(
            self.dataset_name
        )

        x_train_lime, y_train_lime, x_test_lime, y_test_lime, feature_names, categorical_names, column_transformer = \
            get_data(self.dataset_name, for_lime=True)

        categorical_features = list(categorical_names.keys()) if categorical_names is not None else None
        self.lime_instance = LimeTabularExplainer(x_train_lime,
                                                  feature_selection='none',
                                                  discretize_continuous=False,
                                                  categorical_names=categorical_names,
                                                  categorical_features=categorical_features,
                                                  feature_names=feature_names,
                                                  verbose=False,
                                                  mode='regression' if self.regression else 'classification')

        print('\nSingle-task MLP baseline running...')
        stl_model = None
        for _ in tqdm(range(1, self.runs + 1)):
            self.stl_model_trainer = Trainer(
                model=mlp_baseline(self.regression) if not self.tune_arch else None,
                surrogate=None, x_train=x_train, y_train=y_train,
                regression=self.regression
            )

            stl_model = self.stl_model_trainer.train(epochs=self.stl_epochs, tune=self.tune_arch,
                                                     verbose=self.verbose,
                                                     es_patience=self.es_patience, pl_patience=self.pl_patience)
            global_surrogate = self.stl_model_trainer.train_surrogate(stl_model)

            self.stl_model_evaluator = Evaluator(black_box=stl_model, surrogate=global_surrogate,
                                                 x_test=x_test, y_test=y_test)

            stl_test_fid.append(self.stl_model_evaluator.fidelity(sklearn_surrogate=True))
            stl_test_r2_models.append(self.stl_model_evaluator.r2_agreement(sklearn_surrogate=True))

            score1 = self.stl_model_evaluator.mse() if self.regression else self.stl_model_evaluator.accuracy()
            stl_test_scores1.append(score1)
            score2 = self.stl_model_evaluator.r2() if self.regression else self.stl_model_evaluator.f1()
            stl_test_scores2.append(score2)

        stl_score = np.mean(stl_test_scores1)
        print(
            f'\nSTL (mean) {"MSE" if self.regression else "Accuracy"} = {stl_score}, Std. = {np.std(stl_test_scores1)}')
        stl_fid = np.mean(stl_test_fid)
        print(f'STL (mean) Fidelity = {stl_fid}, Std. = {np.std(stl_test_fid)}')
        print(
            f'STL (mean) '
            f'{"R^2" if self.regression else "F1"} = {np.mean(stl_test_scores2)}, Std. = {np.std(stl_test_scores2)}'
        )
        print(f'STL (mean) R^2 BB-Surrogate = {np.mean(stl_test_r2_models)}, Std. = {np.std(stl_test_r2_models)}')

        if self.regression:
            def stl_lime_predict_fn(x):
                return stl_model.predict(
                    column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
            # stl_lime_predict_fn = lambda x: stl_model.predict(
            #     column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
        else:
            def stl_lime_predict_fn(x):
                return np.squeeze(
                    np.array([1 - stl_model.predict(column_transformer.transform(x)
                                                    if column_transformer is not None else x, verbose=0).flatten(),
                              stl_model.predict(
                                  column_transformer.transform(x) if column_transformer is not None else x,
                                  verbose=0).flatten()])
                ).T
            # stl_lime_predict_fn = lambda x: np.squeeze(
            #     np.array([1 - stl_model.predict(column_transformer.transform(x)
            #                                     if column_transformer is not None else x, verbose=0).flatten(),
            #               stl_model.predict(column_transformer.transform(x) if column_transformer is not None else x,
            #                                 verbose=0).flatten()])
            # ).T

        x_test_lime_sample = None
        if self.dataset_name == 'adult' or self.dataset_name == 'housing':
            x_test_lime_sample = x_test_lime[np.random.choice(x_test_lime.shape[0], 500, replace=False), :]
            print('LIME test sample shape:', x_test_lime_sample.shape)
        stl_local_fid = self.stl_model_evaluator.local_fidelity(
            train_points=x_train_lime,
            test_points=x_test_lime_sample if self.dataset_name == 'housing' or self.dataset_name == 'adult'
            else x_test_lime, column_transformer=column_transformer,
            e=self.lime_instance,
            predict_function=stl_lime_predict_fn, categorical_features=categorical_features
        )
        print(f'STL Local Fidelity =', stl_local_fid)

        return stl_model, stl_score, stl_fid, global_surrogate

    def run_mtl(self, stl_model, stl_score, stl_fid):
        x_train, y_train, x_test, y_test, _, _, _ = get_data(
            self.dataset_name
        )

        x_train_lime, y_train_lime, x_test_lime, y_test_lime, feature_names, categorical_names, column_transformer = \
            get_data(self.dataset_name, for_lime=True)

        x_test_lime_sample = None
        if self.dataset_name == 'adult' or self.dataset_name == 'housing':
            x_test_lime_sample = x_test_lime[np.random.choice(x_test_lime.shape[0], 500, replace=False), :]
            print('LIME test sample shape:', x_test_lime_sample.shape)

        categorical_features = list(categorical_names.keys()) if categorical_names is not None else None

        # run MTL
        print('\nMTL training running...')
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        mtl_mean_test_fid = list()
        mtl_mean_test_scores1 = list()
        mtl_mean_test_scores2 = list()
        mtl_mean_test_r2_models = list()
        mtl_mean_test_r2_w_stl = list()

        mtl_std_test_fid = list()
        mtl_std_test_scores1 = list()
        mtl_std_test_scores2 = list()
        mtl_std_test_r2_models = list()
        mtl_std_test_r2_w_stl = list()

        mtl_test_local_fid = list()

        for alpha in tqdm(alphas):
            # size = runs
            mtl_test_fid = list()
            mtl_test_scores1, mtl_test_scores2 = list(), list()
            mtl_test_r2_w_stl, mtl_test_r2_models = list(), list()
            for _ in range(1, self.runs + 1):
                mtl_model = SurrogateMTL(black_box=tf.keras.models.clone_model(stl_model),
                                         regression=self.regression, alpha=alpha)
                self.mtl_model_trainer = Trainer(
                    model=mtl_model, surrogate=None, regression=self.regression,
                    x_train=x_train, y_train=y_train
                )
                mtl_model = self.mtl_model_trainer.train(epochs=self.mtl_epochs, tune=False,
                                                         verbose=self.verbose,
                                                         es_patience=self.es_patience, pl_patience=self.pl_patience)

                if self.regression:
                    def mtl_lime_predict_fn(x):
                        return mtl_model.black_box.predict(
                            column_transformer.transform(x) if column_transformer is not None else x,
                            verbose=0).flatten()
                    # mtl_lime_predict_fn = lambda x: mtl_model.black_box.predict(
                    #     column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
                else:
                    def mtl_lime_predict_fn(x):
                        return np.squeeze(
                            np.array([1 - mtl_model.black_box.predict(column_transformer.transform(x)
                                                                      if column_transformer is not None else x,
                                                                      verbose=0).flatten(),
                                      mtl_model.black_box.predict(
                                          column_transformer.transform(x) if column_transformer is not None else x,
                                          verbose=0).flatten()])
                        ).T
                    # mtl_lime_predict_fn = lambda x: np.squeeze(
                    #     np.array([1 - mtl_model.black_box.predict(column_transformer.transform(x)
                    #                                               if column_transformer is not None else x,
                    #                                               verbose=0).flatten(),
                    #               mtl_model.black_box.predict(
                    #                   column_transformer.transform(x) if column_transformer is not None else x,
                    #                   verbose=0).flatten()])
                    # ).T

                self.mtl_model_evaluator = Evaluator(black_box=mtl_model.black_box, surrogate=mtl_model.explainer,
                                                     x_test=x_test, y_test=y_test)

                score1 = self.mtl_model_evaluator.mse() if self.regression else self.mtl_model_evaluator.accuracy()
                mtl_test_scores1.append(score1)
                score2 = self.mtl_model_evaluator.r2() if self.regression else self.mtl_model_evaluator.f1()
                mtl_test_scores2.append(score2)
                fid = self.mtl_model_evaluator.fidelity(sklearn_surrogate=False)

                mtl_test_fid.append(fid)
                mtl_test_r2_models.append(self.mtl_model_evaluator.r2_agreement())
                mtl_test_r2_w_stl.append(Evaluator(black_box=mtl_model.black_box, surrogate=stl_model,
                                                   x_test=x_test, y_test=y_test).r2_agreement())

                # local_fid = self.mtl_model_evaluator.local_fidelity(train_points=x_train,
                #                                                     test_points=x_test_lime_sample
                #                                                     if self.dataset_name == 'housing' or
                #                                                     self.dataset_name == 'adult' else x_test_lime,
                #                                                     e=self.lime_instance,
                #                                                     categorical_features=categorical_features,
                #                                                     predict_function=mtl_lime_predict_fn,
                #                                                     use_tqdm=False, )
                #
                # print(
                #     f'Alpha = {alpha}, MSE = {self.mtl_model_evaluator.mse()}, Local Fidelity = {local_fid}')

            # size = len(alphas)
            mtl_mean_test_fid.append(np.mean(mtl_test_fid))
            mtl_std_test_fid.append(np.std(mtl_test_fid))
            mtl_mean_test_scores1.append(np.mean(mtl_test_scores1))
            mtl_std_test_scores1.append(np.std(mtl_test_scores1))
            mtl_mean_test_scores2.append(np.mean(mtl_test_scores2))
            mtl_std_test_scores2.append(np.std(mtl_test_scores2))
            mtl_mean_test_r2_models.append(np.mean(mtl_test_r2_models))
            mtl_std_test_r2_models.append(np.std(mtl_test_r2_models))
            mtl_mean_test_r2_w_stl.append(np.mean(mtl_test_r2_w_stl))
            mtl_std_test_r2_w_stl.append(np.std(mtl_test_r2_w_stl))

            local_fid = self.mtl_model_evaluator.local_fidelity(train_points=x_train_lime,
                                                                test_points=x_test_lime_sample
                                                                if self.dataset_name == 'housing'
                                                                or self.dataset_name == 'adult' else x_test_lime,
                                                                e=self.lime_instance,
                                                                column_transformer=column_transformer,
                                                                categorical_features=categorical_features,
                                                                predict_function=mtl_lime_predict_fn, use_tqdm=False, )
            mtl_test_local_fid.append(local_fid)

        full_runs = pd.DataFrame({'Alphas': alphas, 'Mean Fidelity': mtl_mean_test_fid,
                                  'Std. Fidelity': mtl_std_test_fid,
                                  f'Mean {"MSE" if self.regression else "Accuracy"}': mtl_mean_test_scores1,
                                  f'Std. {"MSE" if self.regression else "Accuracy"}': mtl_std_test_scores1,
                                  f'Mean {"R^2" if self.regression else "F1"}': mtl_mean_test_scores2,
                                  f'Std. {"R^2" if self.regression else "F1"}': mtl_std_test_scores2,
                                  'Mean R^2 BB-Surrogate': mtl_mean_test_r2_models,
                                  'Std. R^2 BB-Surrogate': mtl_std_test_r2_models,
                                  'Mean R^2 w/ STL BB': mtl_mean_test_r2_w_stl,
                                  'Std. R^2 w/ STL BB': mtl_std_test_r2_w_stl,
                                  'Local Fidelity (LIME)': mtl_test_local_fid
                                  })

        if self.show_full_scores:
            print(full_runs)

        if self.save_plots:
            self.plot(full_runs, stl_score, stl_fid, mtl_mean_test_scores1, mtl_mean_test_fid)

    def run_decoupled(self, stl_model):
        print('\nDecoupled MTL training running...')

        # size = runs
        decoupled_mtl_test_fid = list()
        decoupled_mtl_test_scores1 = list()
        decoupled_mtl_test_scores2 = list()
        decoupled_mtl_test_r2_models = list()
        decoupled_mtl_test_r2_w_stl = list()

        x_train, y_train, x_test, y_test, _, _, _ = get_data(
            self.dataset_name
        )

        for _ in range(1, self.runs + 1):
            decoupled_mtl_model = SurrogateMTLDecoupled(black_box=tf.keras.models.clone_model(stl_model),
                                                        regression=self.regression)
            decoupled_mtl_model_trainer = Trainer(
                model=decoupled_mtl_model, surrogate=None, regression=self.regression,
                x_train=x_train, y_train=y_train
            )
            decoupled_mtl_model = decoupled_mtl_model_trainer.train(epochs=self.mtl_epochs, tune=False,
                                                                    verbose=self.verbose,
                                                                    es_patience=self.es_patience,
                                                                    es_metric='val_clf_loss',
                                                                    pl_patience=self.pl_patience,
                                                                    pl_metric='val_clf_loss')

            decoupled_mtl_model_evaluator = Evaluator(black_box=decoupled_mtl_model.black_box,
                                                      surrogate=decoupled_mtl_model.explainer,
                                                      x_test=x_test, y_test=y_test)

            score1 = decoupled_mtl_model_evaluator.mse() if \
                self.regression else decoupled_mtl_model_evaluator.accuracy()
            decoupled_mtl_test_scores1.append(score1)
            score2 = decoupled_mtl_model_evaluator.r2() if self.regression else decoupled_mtl_model_evaluator.f1()
            decoupled_mtl_test_scores2.append(score2)
            fid = decoupled_mtl_model_evaluator.fidelity(sklearn_surrogate=False)

            decoupled_mtl_test_fid.append(fid)
            decoupled_mtl_test_r2_models.append(decoupled_mtl_model_evaluator.r2_agreement())
            decoupled_mtl_test_r2_w_stl.append(Evaluator(black_box=decoupled_mtl_model.black_box, surrogate=stl_model,
                                                         x_test=x_test, y_test=y_test).r2_agreement())

            full_runs = pd.DataFrame({'Mean Fidelity': np.mean(decoupled_mtl_test_fid),
                                      'Std. Fidelity': np.std(decoupled_mtl_test_fid),
                                      f'Mean {"MSE" if self.regression else "Accuracy"}':
                                          np.mean(decoupled_mtl_test_scores1),
                                      f'Std. {"MSE" if self.regression else "Accuracy"}':
                                          np.std(decoupled_mtl_test_scores1),
                                      f'Mean {"R^2" if self.regression else "F1"}': np.mean(decoupled_mtl_test_scores2),
                                      f'Std. {"R^2" if self.regression else "F1"}': np.std(decoupled_mtl_test_scores2),
                                      'Mean R^2 BB-Surrogate': np.mean(decoupled_mtl_test_r2_models),
                                      'Std. R^2 BB-Surrogate': np.std(decoupled_mtl_test_r2_models),
                                      'Mean R^2 w/ STL BB': np.mean(decoupled_mtl_test_r2_w_stl),
                                      'Std. R^2 w/ STL BB': np.std(decoupled_mtl_test_r2_w_stl),
                                      }, index=[0])

            if self.show_full_scores:
                print('\nDecoupled results:\n')
                print(full_runs)

    def run_decoupled_mtl(self, stl_model):
        print('\nDecoupled MTL with distinct losses training running...')
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        x_train, y_train, x_test, y_test, _, _, _ = get_data(
            self.dataset_name
        )

        mtl_mean_test_fid = list()
        mtl_mean_test_scores1 = list()
        mtl_mean_test_scores2 = list()
        mtl_mean_test_r2_models = list()
        mtl_mean_test_r2_w_stl = list()

        mtl_std_test_fid = list()
        mtl_std_test_scores1 = list()
        mtl_std_test_scores2 = list()
        mtl_std_test_r2_models = list()
        mtl_std_test_r2_w_stl = list()

        for alpha in tqdm(alphas):
            # size = runs
            mtl_test_fid = list()
            mtl_test_scores1, mtl_test_scores2 = list(), list()
            mtl_test_r2_w_stl, mtl_test_r2_models = list(), list()
            for _ in range(1, self.runs + 1):
                decoupled_mtl_model = SurrogateMTLDecoupledV2(black_box=tf.keras.models.clone_model(stl_model),
                                                              regression=self.regression, alpha=alpha)
                self.mtl_model_trainer = Trainer(
                    model=decoupled_mtl_model, surrogate=None, regression=self.regression,
                    x_train=x_train, y_train=y_train
                )
                decoupled_mtl_model = self.mtl_model_trainer.train(epochs=self.mtl_epochs, tune=False,
                                                                   verbose=self.verbose,
                                                                   es_patience=self.es_patience,
                                                                   pl_patience=self.pl_patience)

                self.mtl_model_evaluator = Evaluator(black_box=decoupled_mtl_model.black_box,
                                                     surrogate=decoupled_mtl_model.explainer,
                                                     x_test=x_test, y_test=y_test)

                score1 = self.mtl_model_evaluator.mse() if self.regression else self.mtl_model_evaluator.accuracy()
                mtl_test_scores1.append(score1)
                score2 = self.mtl_model_evaluator.r2() if self.regression else self.mtl_model_evaluator.f1()
                mtl_test_scores2.append(score2)
                fid = self.mtl_model_evaluator.fidelity(sklearn_surrogate=False)

                mtl_test_fid.append(fid)
                mtl_test_r2_models.append(self.mtl_model_evaluator.r2_agreement())
                mtl_test_r2_w_stl.append(Evaluator(black_box=decoupled_mtl_model.black_box, surrogate=stl_model,
                                                   x_test=x_test, y_test=y_test).r2_agreement())

            # size = len(alphas)
            mtl_mean_test_fid.append(np.mean(mtl_test_fid))
            mtl_std_test_fid.append(np.std(mtl_test_fid))
            mtl_mean_test_scores1.append(np.mean(mtl_test_scores1))
            mtl_std_test_scores1.append(np.std(mtl_test_scores1))
            mtl_mean_test_scores2.append(np.mean(mtl_test_scores2))
            mtl_std_test_scores2.append(np.std(mtl_test_scores2))
            mtl_mean_test_r2_models.append(np.mean(mtl_test_r2_models))
            mtl_std_test_r2_models.append(np.std(mtl_test_r2_models))
            mtl_mean_test_r2_w_stl.append(np.mean(mtl_test_r2_w_stl))
            mtl_std_test_r2_w_stl.append(np.std(mtl_test_r2_w_stl))

        full_runs = pd.DataFrame({'Alpha': alphas, 'Mean Fidelity': mtl_mean_test_fid,
                                  'Std. Fidelity': mtl_std_test_fid,
                                  f'Mean {"MSE" if self.regression else "Accuracy"}': mtl_mean_test_scores1,
                                  f'Std. {"MSE" if self.regression else "Accuracy"}': mtl_std_test_scores1,
                                  f'Mean {"R^2" if self.regression else "F1"}': mtl_mean_test_scores2,
                                  f'Std. {"R^2" if self.regression else "F1"}': mtl_std_test_scores2,
                                  'Mean R^2 BB-Surrogate': mtl_mean_test_r2_models,
                                  'Std. R^2 BB-Surrogate': mtl_std_test_r2_models,
                                  'Mean R^2 w/ STL BB': mtl_mean_test_r2_w_stl,
                                  'Std. R^2 w/ STL BB': mtl_std_test_r2_w_stl,
                                  })

        if self.show_full_scores:
            print('\nDecoupled v2 results:\n')
            print(full_runs)

    def run_regularized(self, stl_model, stl_surrogate=None):
        print('\nMTL with distillation training running...')

        # lambdas = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 2.0, 5.0]
        lambdas = [0]

        x_train, y_train, x_test, y_test, _, _, _ = get_data(
            self.dataset_name
        )

        mtl_mean_test_fid = list()
        mtl_mean_test_scores1 = list()
        mtl_mean_test_scores2 = list()
        mtl_mean_test_r2_models = list()
        mtl_mean_test_r2_w_stl = list()

        mtl_std_test_fid = list()
        mtl_std_test_scores1 = list()
        mtl_std_test_scores2 = list()
        mtl_std_test_r2_models = list()
        mtl_std_test_r2_w_stl = list()

        for lambda_ in tqdm(lambdas):
            mtl_test_fid = list()
            mtl_test_scores1, mtl_test_scores2 = list(), list()
            mtl_test_r2_w_stl, mtl_test_r2_models = list(), list()
            for _ in range(1, self.runs + 1):
                distilled_model = SurrogateMTLRegularized(black_box=stl_model, regression=self.regression,
                                                          lambda_=lambda_, alpha=0.5)
                distilled_model.check_init_perf(x_test, y_test, regression=self.regression)  # check initial performance
                distilled_model.g.build(input_shape=(1, x_train.shape[1]))
                if stl_surrogate is not None:
                    g_init_params = [np.expand_dims(np.array(stl_surrogate.coef_), axis=-1),
                                     np.expand_dims(np.array(stl_surrogate.intercept_), axis=-1)]
                    distilled_model.g.set_weights(g_init_params)

                self.mtl_model_trainer = Trainer(
                    model=distilled_model, surrogate=None, regression=self.regression,
                    x_train=x_train, y_train=y_train
                )

                optimizers = [
                    tf.keras.optimizers.Adam(learning_rate=5e-5),
                    tf.keras.optimizers.Adam()
                ]
                optimizers_and_layers = [(optimizers[0], distilled_model.f_prime),
                                         (optimizers[1], distilled_model.g)]
                optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
                distilled_model: tf.keras.Model = self.mtl_model_trainer.train(epochs=self.mtl_epochs, tune=False,
                                                                               verbose=self.verbose, opt=optimizer,
                                                                               es_patience=self.es_patience,
                                                                               pl_patience=self.pl_patience)

                self.mtl_model_evaluator = Evaluator(black_box=distilled_model.f_prime,
                                                     surrogate=distilled_model.g,
                                                     x_test=x_test, y_test=y_test)

                score1 = self.mtl_model_evaluator.mse() if self.regression else self.mtl_model_evaluator.accuracy()
                mtl_test_scores1.append(score1)
                score2 = self.mtl_model_evaluator.r2() if self.regression else self.mtl_model_evaluator.f1()
                mtl_test_scores2.append(score2)
                fid = self.mtl_model_evaluator.fidelity(sklearn_surrogate=False)

                mtl_test_fid.append(fid)
                mtl_test_r2_models.append(self.mtl_model_evaluator.r2_agreement())
                mtl_test_r2_w_stl.append(Evaluator(black_box=distilled_model.f_prime, surrogate=stl_model,
                                                   x_test=x_test, y_test=y_test).r2_agreement())

            # size = len(lambdas)
            mtl_mean_test_fid.append(np.mean(mtl_test_fid))
            mtl_std_test_fid.append(np.std(mtl_test_fid))
            mtl_mean_test_scores1.append(np.mean(mtl_test_scores1))
            mtl_std_test_scores1.append(np.std(mtl_test_scores1))
            mtl_mean_test_scores2.append(np.mean(mtl_test_scores2))
            mtl_std_test_scores2.append(np.std(mtl_test_scores2))
            mtl_mean_test_r2_models.append(np.mean(mtl_test_r2_models))
            mtl_std_test_r2_models.append(np.std(mtl_test_r2_models))
            mtl_mean_test_r2_w_stl.append(np.mean(mtl_test_r2_w_stl))
            mtl_std_test_r2_w_stl.append(np.std(mtl_test_r2_w_stl))

        full_runs = pd.DataFrame({'Lambda': lambdas, 'Mean Fidelity': mtl_mean_test_fid,
                                  'Std. Fidelity': mtl_std_test_fid,
                                  f'Mean {"MSE" if self.regression else "Accuracy"}': mtl_mean_test_scores1,
                                  f'Std. {"MSE" if self.regression else "Accuracy"}': mtl_std_test_scores1,
                                  f'Mean {"R^2" if self.regression else "F1"}': mtl_mean_test_scores2,
                                  f'Std. {"R^2" if self.regression else "F1"}': mtl_std_test_scores2,
                                  'Mean R^2 BB-Surrogate': mtl_mean_test_r2_models,
                                  'Std. R^2 BB-Surrogate': mtl_std_test_r2_models,
                                  'Mean R^2 w/ STL BB': mtl_mean_test_r2_w_stl,
                                  'Std. R^2 w/ STL BB': mtl_std_test_r2_w_stl,
                                  })

        if self.show_full_scores:
            print('\nDistilled results:\n')
            print(full_runs)
