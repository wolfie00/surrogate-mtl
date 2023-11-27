from models import *
from data import *
from tqdm import tqdm
from trainer import Trainer
from evaluator import Evaluator
from lime.lime_tabular import LimeTabularExplainer


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

        self.stl_model_trainer, self.mtl_model_trainer = None, None
        self.stl_model_evaluator, self.mtl_model_evaluator = None, None

    def run(self):

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

        # run linear baseline
        print('\nLinear baseline training...')
        linear_bsl_trainer = Trainer(
            model=linear_baseline(self.regression), surrogate=None, x_train=x_train, y_train=y_train,
            regression=self.regression
        )

        linear_bsl = linear_bsl_trainer.train(epochs=self.stl_epochs, tune=self.tune_arch,
                                              verbose=self.verbose)
        linear_bsl_evaluator = Evaluator(black_box=linear_bsl, surrogate=None,
                                         x_test=x_test, y_test=y_test)
        linear_bsl_evaluation = linear_bsl_evaluator.mse() if self.regression else linear_bsl_evaluator.accuracy()
        print(f'\nTest {"MSE" if self.regression else "Accuracy"} score of linear baseline:',
              linear_bsl_evaluation)

        print('\n*************************************************************************************')

        stl_test_fid = list()
        stl_test_r2_models = list()

        stl_test_scores1, stl_test_scores2 = list(), list()

        print('\nSingle-task MLP baseline running...')
        stl_model = None
        for _ in tqdm(range(1, self.runs + 1)):
            self.stl_model_trainer = Trainer(
                model=mlp_baseline(self.regression), surrogate=None, x_train=x_train, y_train=y_train,
                regression=self.regression
            )

            stl_model = self.stl_model_trainer.train(epochs=self.stl_epochs, tune=self.tune_arch,
                                                     verbose=self.verbose,
                                                     es_patience=self.es_patience, pl_patience=self.pl_patience)
            global_surrogate = linear_bsl_trainer.train_surrogate(stl_model)

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
            def stl_lime_predict_fn(x): return stl_model.predict(
                column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
            # stl_lime_predict_fn = lambda x: stl_model.predict(
            #     column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
        else:
            def stl_lime_predict_fn(x): return np.squeeze(
                np.array([1 - stl_model.predict(column_transformer.transform(x)
                                                if column_transformer is not None else x, verbose=0).flatten(),
                          stl_model.predict(column_transformer.transform(x) if column_transformer is not None else x,
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
        stl_local_fid = self.stl_model_evaluator.local_fidelity(
            train_points=x_train,
            test_points=x_test_lime_sample if self.dataset_name == 'housing' or self.dataset_name == 'adult'
            else x_test_lime,
            e=self.lime_instance,
            predict_function=stl_lime_predict_fn, categorical_features=categorical_features
        )
        print(f'STL Local Fidelity =', stl_local_fid)

        print('\n*************************************************************************************')

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
                mtl_model = self.mtl_model_trainer.train(epochs=self.mtl_epochs, tune=self.tune_arch,
                                                         verbose=self.verbose,
                                                         es_patience=self.es_patience, pl_patience=self.pl_patience)

                if self.regression:
                    def mtl_lime_predict_fn(x): return mtl_model.black_box.predict(
                        column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
                    # mtl_lime_predict_fn = lambda x: mtl_model.black_box.predict(
                    #     column_transformer.transform(x) if column_transformer is not None else x, verbose=0).flatten()
                else:
                    def mtl_lime_predict_fn(x): return np.squeeze(
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

            local_fid = self.mtl_model_evaluator.local_fidelity(train_points=x_train,
                                                                test_points=x_test_lime_sample
                                                                if self.dataset_name == 'housing'
                                                                or self.dataset_name == 'adult' else x_test_lime,
                                                                e=self.lime_instance,
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

        # mtl_mean_local_test_fid = list()
        # mtl_std_local_test_fid = list()
        # mtl_mean_local_test_mse = list()
        # mtl_std_local_test_mse = list()
        # for alpha in tqdm(alphas):
        #     mtl_local_test_fid = list()
        #     mtl_local_test_mse = list()
        #     for run in range(1, 1 + 1):
        #         mtl_local_model = SurrogateMTL(black_box=tf.keras.models.clone_model(stl_model),
        #                                        regression=self.regression,
        #                                        alpha=alpha)
        #
        #         mtl_local_model_trainer = Trainer(
        #             model=mtl_local_model, surrogate=None, x_train=x_train, y_train=y_train,
        #             regression=self.regression
        #         )
        #
        #         mtl_local_model = mtl_local_model_trainer.train(epochs=self.mtl_epochs, tune=self.tune_arch,
        #                                                         verbose=self.verbose, es_patience=self.es_patience,
        #                                                         pl_patience=self.pl_patience)
        #
        #         mtl_local_model_evaluator = Evaluator(black_box=mtl_local_model.black_box,
        #                                               surrogate=mtl_local_model.explainer,
        #                                               x_test=x_test, y_test=y_test)
        #
        #         local_fid = mtl_local_model_evaluator.local_fidelity(x_train,
        #                                                              x_test_lime_sample
        #                                                              if self.dataset_name == 'housing'
        #                                                              or self.dataset_name == 'adult' else x_test_lime,
        #                                                              e=self.lime_instance, use_tqdm=False,
        #                                                              categorical_features=categorical_features)
        #         print(
        #             f'Alpha = {alpha}, MSE = {mtl_local_model_evaluator.mse()}, Local Fidelity = {local_fid}')
        #         mtl_local_test_fid.append(local_fid)
        #         mtl_local_test_mse.append(mtl_local_model_evaluator.mse())

            # mean_mtl_local_test_fidelities.append(np.mean(mtl_local_test_fidelities))
            # std_mtl_local_test_fidelities.append(np.std(mtl_local_test_fidelities))
            # mean_mtl_local_test_mse.append(np.mean(mtl_local_test_mse))
            # std_mtl_local_test_mse.append(np.std(mtl_local_test_mse))
