# Specify model
# Code adapted from https://www.tensorflow.org/probability/examples/Linear_Mixed_Effects_Model_Variational_Inference
def make_joint_distribution_coroutine(n_categories, category, fixed_effect_var):

  def model():
    intercept = yield tfd.Normal(loc=0., scale=1., name='intercept')
    weights = yield tfd.Normal(loc=0., scale=tf.ones(len(x_num)), name='weights')
    cat_prior = yield tfd.Normal(
      loc=tf.zeros(n_categories), scale=1., name='cat_prior')
    random_effect = tf.gather(cat_prior, category, axis=-1)
    fixed_effect = intercept + tf.matmul(fixed_effect_vars, tf.expand_dims(weights, axis=-1))
    linear_response = fixed_effect + random_effect
    yield tfd.Normal(loc=linear_response, scale=1., name='likelihood')
  
  return tfd.JointDistributionCoroutineAutoBatched(model)

category = X_train_ct[hicard_var].values
category = tf.cast(category, dtype=tf.int32)
fixed_effect_vars = X_train_ct[x_num].values
fixed_effect_vars = tf.cast(fixed_effect_vars, dtype=tf.float32)
joint = make_joint_distribution_coroutine(n_categories, category, fixed_effect_vars)

# Define a closure over the joint distribution 
# to condition on the observed labels.
def target_log_prob_fn(*args):
  return joint.log_prob(*args, likelihood=y_train)

# Specify surrogate posterior distributions
surrogate_posterior = tfp.experimental.vi.build_factored_surrogate_posterior(
  event_shape=joint.event_shape_tensor()[:-1],)

optimizer = tf.optimizers.Adam(learning_rate=1e-2)
losses = tfp.vi.fit_surrogate_posterior(
    target_log_prob_fn, 
    surrogate_posterior,
    optimizer=optimizer,
    num_steps=3000, 
    seed=42,
    sample_size=2)

(intercept_, 
 weights_, 
 cat_weights_), _ = surrogate_posterior.sample_distributions()

# Plot the loss function
fig, ax = plt.subplots(figsize=(10, 3))
_ = ax.plot(losses, 'k-')
ax.set(xlabel="Iteration",
       ylabel="Loss (ELBO)",
       title="Loss during training",
       ylim=0)


# Plot true RE and posterior predictions
cat_counts = (X_train_ct.astype({hicard_var: "int"})
                .groupby(by=["category"], observed=True)
                .agg("size")
                .sort_values(ascending=False)
                .reset_index(name="count"))

means = cat_weights_.mean()
stds = cat_weights_.stddev()
true_u = (pd.concat([
    X_train_ct.reset_index(drop=True), 
    pd.DataFrame(Zu[:n_train], columns=["Zu"])
  ], axis=1)).groupby(by=["category"], observed=True).agg("mean").reset_index()

fig, ax = plt.subplots(figsize=(20, 5))

for idx, row in cat_counts.iterrows():
  mid = means[row.category]
  std = stds[row.category]
  ax.vlines(idx, mid - std, mid + std, linewidth=3)
  _ = ax.plot(idx, means[row.category], 'ko', mfc='w', mew=2, ms=7)
  true = true_u.loc[true_u.category == row.category, "Zu"].values
  _ = ax.plot(idx, true, 'r.', mfc='w', mew=2, ms=7)

_ = ax.set(
    xticks=np.arange(len(cat_counts)),
    xlim=(-1, len(cat_counts)),
    ylabel="Category effect",
    title=r"Predictions of category effects on y (mean $\pm$ 1 std. dev.)",
)
_ = ax.set_xticklabels(cat_counts.category, rotation=90)

# RE predictions plot
p = bop.figure(title="Random Effects Predictions", x_axis_label="Ground truth", y_axis_label="Predictions", width=400, height=400)
re_vi = cat_weights_.mean().numpy()
rev = np.array([float(re_vi[g]) for g in X_train[hicard_var]])
p.circle(Zu[:n_train], rev, color="#9f86c0")
p.add_layout(bom.Slope(gradient=1, y_intercept=0, line_color="black", line_width=2))
bop.show(p)

# y prediction plot
y_pred_train = intercept_.mean() + np.matmul(X_train_ct[x_num].values, weights_.mean()) + rev[:n_train]
y_pred_val = intercept_.mean() + np.matmul(X_val_ct[x_num].values, weights_.mean()) + np.array([float(re_vi[g]) for g in X_val[hicard_var]])
p = plot_from_predictions(y_pred_train, y_true_train, y_pred_val, y_true_val, log_scale=log_plot)