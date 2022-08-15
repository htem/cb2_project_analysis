from gunpowder.jax import GenericJaxModel
import haiku
import optax
import jax
import jax.numpy as jnp

class LinearModel(GenericJaxModel):
    def __init__(self, is_training):
        super().__init__(is_training)

        def _linear(x):
            return haiku.Linear(1, False)(x)

        self.linear = haiku.without_apply_rng(haiku.transform(_linear))
        self.opt = optax.sgd(learning_rate=1e-6, momentum=0.999)

    def initialize(self, rng_key, inputs):
        rng_key = jax.random.PRNGKey(42)
        pattern = inputs['pattern']
        weight = self.linear.init(rng_key, pattern)
        opt_state = self.opt.init(weight)
        return (weight, opt_state)

    def forward(self, params, inputs):
        pattern = inputs['pattern']
        return {'pred': self.linear.apply(params[0], pattern)}

    def _loss_fn(self, weight, pattern, cls):
        pred = self.linear.apply(weight, pattern)
        loss = optax.l2_loss(predictions=pred, targets=cls)*2
        loss_mean = loss.mean()
        return loss_mean, (pred, loss, loss_mean)

    def _apply_optimizer(self, params, grads):
        updates, new_opt_state = self.opt.update(grads, params[1])
        new_weight = optax.apply_updates(params[0], updates)
        return new_weight, new_opt_state

    def train_step(self, params, inputs, pmapped=False):
        pattern = inputs['pattern']
        cls = inputs['cls']

        grads, (pred, loss, loss_mean) = jax.grad(
            self._loss_fn, has_aux=True)(params[0], pattern, cls)

        new_weight, new_opt_state = self._apply_optimizer(params, grads)
        new_params = (new_weight, new_opt_state)

        outputs = {
            'pred': pred,
            'grad': loss,
        }
        return new_params, outputs, loss_mean


class SigmoidPosLinearModel(GenericJaxModel):
    def __init__(self, is_training, lr,
                 weight_range_mult=1.0):
        super().__init__(is_training)

        def _linear(x):
            pred = haiku.Linear(1, False)(x)
            pred = jnp.subtract(pred, 10)
            pred = jax.nn.sigmoid(pred)
            return pred

        self.linear = haiku.without_apply_rng(haiku.transform(_linear))
        # self.opt = optax.sgd(learning_rate=lr, momentum=0.99)
        self.opt = optax.sgd(learning_rate=lr, momentum=0.9)
        self.min_val = -0.5
        self.max_val = 0.5
        self.weight_range_mult = weight_range_mult

    def initialize(self, rng_key, inputs):
        rng_key = jax.random.PRNGKey(42)
        pattern = inputs['pattern']
        weight = self.linear.init(rng_key, pattern)
        opt_state = self.opt.init(weight)
        self.max_val = self.weight_range_mult / len(pattern[0])
        # self.min_val = -self.max_val
        self.min_val = 0
        return (weight, opt_state)

    def forward(self, params, inputs):
        pattern = inputs['pattern']
        pred = self.linear.apply(params[0], pattern)
        return {'pred': pred}

    def _loss_fn(self, weight, pattern, cls):
        pred = self.linear.apply(weight, pattern)
        # loss = optax.l2_loss(predictions=pred, targets=cls)*2
        loss = jnp.abs(cls-pred)
        loss_mean = loss.mean()
        return loss_mean, (pred, loss, loss_mean)

    def _apply_optimizer(self, params, grads):
        updates, new_opt_state = self.opt.update(grads, params[1])
        new_weight = optax.apply_updates(params[0], updates)

        # restrict weights
        new_weight = jax.tree_map(lambda x: jnp.minimum(jnp.maximum(x, self.min_val), self.max_val), new_weight)

        return new_weight, new_opt_state

    def train_step(self, params, inputs, pmapped=False):
        pattern = inputs['pattern']
        cls = inputs['cls']

        grads, (pred, loss, loss_mean) = jax.grad(
            self._loss_fn, has_aux=True)(params[0], pattern, cls)

        new_weight, new_opt_state = self._apply_optimizer(params, grads)
        new_params = (new_weight, new_opt_state)

        outputs = {
            'pred': pred,
            'grad': loss,
        }
        return new_params, outputs, loss_mean


class SigmoidLinearModel(GenericJaxModel):
    def __init__(self, is_training, lr,
                 weight_range_mult=1.0):
        super().__init__(is_training)

        def _linear(x):
            pred = haiku.Linear(1, False)(x)
            # pred = jnp.multiply(pred, 4)  # this makes range = [0, ~40]
            # pred = jnp.subtract(pred, 2)  # this makes range = [-20, 20]
            # pred = jnp.multiply(pred, 16)  # this makes range = [0, ~40]
            pred = jax.nn.sigmoid(pred)
            return pred

        self.linear = haiku.without_apply_rng(haiku.transform(_linear))
        self.opt = optax.sgd(learning_rate=lr, momentum=0.99)
        # self.opt = optax.sgd(learning_rate=lr, momentum=0.9)
        # self.opt = optax.sgd(learning_rate=lr, momentum=0.5)
        # self.opt = optax.sgd(learning_rate=lr)
        self.min_val = -0.5
        self.max_val = 0.5
        self.weight_range_mult = weight_range_mult

    def initialize(self, rng_key, inputs):
        rng_key = jax.random.PRNGKey(42)
        pattern = inputs['pattern']
        weight = self.linear.init(rng_key, pattern)
        opt_state = self.opt.init(weight)
        self.max_val = self.weight_range_mult / len(pattern[0])
        self.min_val = -self.max_val
        return (weight, opt_state)

    def forward(self, params, inputs):
        pattern = inputs['pattern']
        pred = self.linear.apply(params[0], pattern)
        return {'pred': pred}

    def _loss_fn(self, weight, pattern, cls):
        pred = self.linear.apply(weight, pattern)
        # loss = optax.l2_loss(predictions=pred, targets=cls)*2
        loss = jnp.abs(cls-pred)
        loss_mean = loss.mean()
        return loss_mean, (pred, loss, loss_mean)

    def _apply_optimizer(self, params, grads):
        updates, new_opt_state = self.opt.update(grads, params[1])
        new_weight = optax.apply_updates(params[0], updates)

        # restrict weights to [0, 0.1]
        new_weight = jax.tree_map(lambda x: jnp.minimum(jnp.maximum(x, self.min_val), self.max_val), new_weight)

        return new_weight, new_opt_state

    def train_step(self, params, inputs, pmapped=False):
        pattern = inputs['pattern']
        cls = inputs['cls']

        grads, (pred, loss, loss_mean) = jax.grad(
            self._loss_fn, has_aux=True)(params[0], pattern, cls)

        new_weight, new_opt_state = self._apply_optimizer(params, grads)
        new_params = (new_weight, new_opt_state)

        outputs = {
            'pred': pred,
            'grad': loss,
        }
        return new_params, outputs, loss_mean


class SigmoidPosLinearModel2(GenericJaxModel):
    def __init__(self, is_training, lr,
                 act_rate, sigmoid_scale=20,
                 momentum=.99):
        super().__init__(is_training)

        assert act_rate > 0 and act_rate < 1
        self.act_rate = act_rate
        self.sigmoid_scale = sigmoid_scale

        def _linear(x):
            pred = haiku.Linear(1, False)(x)
            pred = jnp.subtract(pred, self.sigmoid_scale/2)
            pred = jax.nn.sigmoid(pred)
            return pred

        self.linear = haiku.without_apply_rng(haiku.transform(_linear))
        self.opt = optax.sgd(learning_rate=lr, momentum=momentum)
        self.min_val = 0
        self.max_val = sigmoid_scale/(.5*act_rate)
        # print(f'scaling: {self.max_val}')

    def initialize(self, rng_key, inputs):
        rng_key = jax.random.PRNGKey(42)
        pattern = inputs['pattern']
        weight = self.linear.init(rng_key, pattern)
        opt_state = self.opt.init(weight)
        self.max_val /= len(pattern[0])
        return (weight, opt_state)

    def forward(self, params, inputs):
        pattern = inputs['pattern']
        pred = self.linear.apply(params[0], pattern)
        return {'pred': pred}

    def _loss_fn(self, weight, pattern, cls):
        pred = self.linear.apply(weight, pattern)
        # loss = optax.l2_loss(predictions=pred, targets=cls)*2
        loss = jnp.abs(cls-pred)
        loss_mean = loss.mean()
        return loss_mean, (pred, loss, loss_mean)

    def _apply_optimizer(self, params, grads):
        updates, new_opt_state = self.opt.update(grads, params[1])
        new_weight = optax.apply_updates(params[0], updates)

        # restrict weights
        new_weight = jax.tree_map(lambda x: jnp.minimum(jnp.maximum(x, self.min_val), self.max_val), new_weight)

        return new_weight, new_opt_state

    def train_step(self, params, inputs, pmapped=False):
        pattern = inputs['pattern']
        cls = inputs['cls']

        grads, (pred, loss, loss_mean) = jax.grad(
            self._loss_fn, has_aux=True)(params[0], pattern, cls)

        new_weight, new_opt_state = self._apply_optimizer(params, grads)
        new_params = (new_weight, new_opt_state)

        outputs = {
            'pred': pred,
            'grad': loss,
        }
        return new_params, outputs, loss_mean




class SigmoidPosLinearModel3(GenericJaxModel):
    def __init__(self, is_training, lr,
                 act_rate, sigmoid_scale=20,
                 momentum=.99):
        super().__init__(is_training)

        assert act_rate > 0 and act_rate < 1
        self.act_rate = act_rate
        self.sigmoid_scale = sigmoid_scale

        def _linear(x):
            pred = haiku.Linear(1, False)(x)
            pred = jnp.subtract(pred, self.sigmoid_scale/2)
            pred = jax.nn.sigmoid(pred)
            return pred

        self.linear = haiku.without_apply_rng(haiku.transform(_linear))
        self.opt = optax.sgd(learning_rate=lr, momentum=momentum)
        self.min_val = 0
        self.max_val = sigmoid_scale/(.5*act_rate)
        print(f'scaling: {self.max_val}')

    def initialize(self, rng_key, inputs):
        rng_key = jax.random.PRNGKey(42)
        pattern = inputs['pattern']
        weight = self.linear.init(rng_key, pattern)
        opt_state = self.opt.init(weight)
        self.max_val /= len(pattern[0])
        weight['linear']['w'] = jnp.add(weight['linear']['w'],
                                        self.max_val/2)
        return (weight, opt_state)

    def forward(self, params, inputs):
        pattern = inputs['pattern']
        pred = self.linear.apply(params[0], pattern)
        return {'pred': pred}

    def _loss_fn(self, weight, pattern, cls):
        pred = self.linear.apply(weight, pattern)
        # loss = optax.l2_loss(predictions=pred, targets=cls)*2
        loss = jnp.abs(cls-pred)
        loss_mean = loss.mean()
        return loss_mean, (pred, loss, loss_mean)

    def _apply_optimizer(self, params, grads):
        updates, new_opt_state = self.opt.update(grads, params[1])
        new_weight = optax.apply_updates(params[0], updates)

        # restrict weights
        new_weight = jax.tree_map(lambda x: jnp.minimum(jnp.maximum(x, self.min_val), self.max_val), new_weight)

        return new_weight, new_opt_state

    def train_step(self, params, inputs, pmapped=False):
        pattern = inputs['pattern']
        cls = inputs['cls']

        grads, (pred, loss, loss_mean) = jax.grad(
            self._loss_fn, has_aux=True)(params[0], pattern, cls)

        new_weight, new_opt_state = self._apply_optimizer(params, grads)
        new_params = (new_weight, new_opt_state)

        outputs = {
            'pred': pred,
            'grad': loss,
        }
        return new_params, outputs, loss_mean



