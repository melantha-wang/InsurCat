# A reduced version of the original code
def reg_nn_lmm(X_train, X_test, y_train, y_test, qs, q_spatial, x_cols, batch_size, epochs, patience, n_neurons, dropout, activation,
        mode, n_sig2bs, n_sig2bs_spatial, est_cors, dist_matrix, spatial_embed_neurons,
        verbose=False, Z_non_linear=False, Z_embed_dim_pct=10, log_params=False, idx=0, shuffle=False, sample_n_train=10000):
    dmatrix_tf = dist_matrix
    X_input = Input(shape=(X_train[x_cols].shape[1],))
    y_true_input = Input(shape=(1,))
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        z_cols = sorted(X_train.columns[X_train.columns.str.startswith('z')].tolist())
        Z_inputs = []
        n_sig2bs_init = len(qs)
        n_RE_inputs = len(qs)
        for _ in range(n_RE_inputs):
            Z_input = Input(shape=(1,), dtype=tf.int64)
            Z_inputs.append(Z_input)
    
    out_hidden = add_layers_functional(X_input, n_neurons, dropout, activation, X_train[x_cols].shape[1])
    y_pred_output = Dense(1)(out_hidden)

    Z_nll_inputs = Z_inputs
    ls = None

    sig2bs_init = np.ones(n_sig2bs_init, dtype=np.float32)
    rhos_init = np.zeros(len(est_cors), dtype=np.float32)
    weibull_init = np.ones(2, dtype=np.float32)
    nll = NLL(mode, 1.0, sig2bs_init, rhos_init, weibull_init, est_cors, Z_non_linear, dmatrix_tf)(
        y_true_input, y_pred_output, Z_nll_inputs)
    model = Model(inputs=[X_input, y_true_input] + Z_inputs, outputs=nll)

    model.compile(optimizer='adam')

    patience = epochs if patience is None else patience
    callbacks = [EarlyStopping(patience=patience, monitor='val_loss')]
    if not Z_non_linear:
        X_train.sort_values(by=z_cols, inplace=True)
        y_train = y_train[X_train.index]
    X_train_z_cols = [X_train[z_col] for z_col in z_cols]
    X_test_z_cols = [X_test[z_col] for z_col in z_cols]
    history = model.fit([X_train[x_cols], y_train] + X_train_z_cols, None,
                        batch_size=batch_size, epochs=epochs, validation_split=0.1,
                        callbacks=callbacks, verbose=verbose, shuffle=shuffle)

    sig2e_est, sig2b_ests, rho_ests, weibull_ests = model.layers[-1].get_vars()
    sig2b_spatial_ests = []
    y_pred_tr = model.predict(
        [X_train[x_cols], y_train] + X_train_z_cols).reshape(X_train.shape[0])
    b_hat = calc_b_hat(X_train, y_train, y_pred_tr, qs, q_spatial, sig2e_est, sig2b_ests, sig2b_spatial_ests,
                Z_non_linear, model, ls, mode, rho_ests, est_cors, dist_matrix, weibull_ests, sample_n_train)
    dummy_y_test = np.random.normal(size=y_test.shape)
    if mode in ['intercepts', 'glmm', 'spatial', 'spatial_and_categoricals']:
        if Z_non_linear or len(qs) > 1:
            Z_tests = []
            for k, q in enumerate(qs):
                Z_test = get_dummies(X_test['z' + str(k)], q)
                Z_tests.append(Z_test)
            Z_test = sparse.hstack(Z_tests)
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0]) + Z_test @ b_hat
        else:
            y_pred = model.predict([X_test[x_cols], dummy_y_test] + X_test_z_cols).reshape(
                X_test.shape[0]) + b_hat[X_test['z0']]
    return model, y_pred, (sig2e_est, list(sig2b_ests), list(sig2b_spatial_ests)), list(rho_ests), list(weibull_ests), history.history
