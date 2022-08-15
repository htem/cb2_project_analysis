import jax


def eval220316(network, params, source, batch_size, seed):

    inputs = source.get(batch_size, seed)
    x, gt = inputs['pattern'], inputs['cls']
    y = jax.jit(network.forward)(params, {'pattern': x})['pred']
    acc = 0
    for yy, zz in zip(y, gt):
        if (yy[0] > 0.5) == bool(zz[0]):
            acc += 1
    acc = float(acc) / len(y)
    print(f"accuracy: {acc}")
    return acc


def eval220317(network, params, source, batch_size, log_dir, seed):

    inputs = source.get(batch_size, seed)
    x, gt = inputs['pattern'], inputs['cls']
    y = jax.jit(network.forward)(params, {'pattern': x})['pred']
    acc = 0
    for yy, zz in zip(y, gt):
        if (yy[0] > 0.5) == bool(zz[0]):
            acc += 1
    acc = float(acc) / len(y)
    print(f"accuracy: {acc}")
    with open(f'{log_dir}/results', 'a') as f:
        f.write(','.join([str(float(yy[0])) for yy in y]))
        f.write('\n')
        f.write(','.join([str(zz[0]) for zz in gt]))
        f.write('\n')
        f.write('acc: ' + str(acc))
        f.write('\n')

    return acc
