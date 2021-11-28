import random
import numpy as np
import copy

def generate_patterns(
        pattern_len,
        count,
        f=.5,
        type='uniform',
        seed=1,
    ):

    if type == 'binary':
        return generate_binary_patterns(pattern_len, count, f, seed=seed)
    patterns = []
    np.random.seed(seed)
    random.seed(seed)
    for i in range(count):
        if type == 'uniform':
            b = [None]*pattern_len
            for k in range(pattern_len):
                b[k] = random.random()
        elif type == 'uniform255':
            rng = np.random.default_rng()
            b = rng.integers(low=0, high=256, size=pattern_len).astype(np.uint8)
        elif type == 'gaussian':
            mu, sigma = 0.5, 0.2 # mean and standard deviation
            b = np.random.normal(mu, sigma, pattern_len)
        output = random.randint(0, 1)
        patterns.append((b, output))
    # print(patterns)
    return patterns

def generate_binary_patterns(pattern_len, count, f, seed=1):
    patterns = []
    np.random.seed(seed)
    random.seed(seed)
    threshold = int(pattern_len*f+0.5)
    base = np.zeros(pattern_len, dtype=np.uint8)
    base[0:threshold] = 1
    for i in range(count):
        np.random.shuffle(base)
        b = base.copy()
        output = random.randint(0, 1)
        patterns.append((b, output))
    # print(patterns)
    return patterns

def add_noise_to_patterns(patterns, type, **kwargs):
    if type == 'uniform':
        return add_noise_float_patterns(
            patterns,
            **kwargs)
    elif type == 'uniform255':
        return add_noise_uniform255_patterns(
            patterns,
            **kwargs)
    elif type == 'binary':
        return add_noise_binary_patterns(
            patterns,
            **kwargs)
    else:
        assert False

def add_noise_binary_patterns(patterns, prob, f=None, n=1,
        small_feature_mode=False,
        noise_mask=None,
        invert_noise_mask=False,
        seed=None,
        mf_mask=None,
        no_adjust_noise_ratio=False,
        ):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    user_noise_mask = noise_mask
    if invert_noise_mask:
        user_noise_mask = np.logical_not(user_noise_mask)
    pattern_len = len(patterns[0][0])
    if f is None:
        f = sum(patterns[0][0]) / pattern_len

    if mf_mask is not None:
        ratio = sum(mf_mask)/len(mf_mask)
        if not no_adjust_noise_ratio:
            prob /= ratio
            # assert prob <= 1
            prob = min(1, prob)
        mf_mask = np.array(mf_mask, dtype=np.uint8)

    # print(f'prob: {prob}')
    # print(f'f: {f}')
    noise_mask = np.zeros(pattern_len, dtype=np.uint8)
    noise_mask[0:int(prob*pattern_len)] = 1
    np.random.shuffle(noise_mask)
    new_random_pattern = np.zeros(pattern_len, dtype=np.uint8)
    new_random_pattern[0:int(f*pattern_len)] = 1
    ret = []

    for pattern in patterns:
        pattern, output = pattern
        for i in range(n):
            if not small_feature_mode:
                np.random.shuffle(noise_mask)
            np.random.shuffle(new_random_pattern)
            mask = noise_mask
            if user_noise_mask is not None:
                mask = np.bitwise_and(noise_mask, user_noise_mask)
            if mf_mask is not None:
                mask = np.bitwise_and(mask, mf_mask)
            # np.where syntax: np.where(condition, x, y) -> x if c else y
            noisy_pattern = np.where(mask, new_random_pattern, pattern)
            # noisy_pattern = np.where(mask, pattern, new_random_pattern)  # wrong
            # noisy_pattern = np.where(new_random_pattern, mask, pattern)  # wrong
            ret.append((noisy_pattern, output))
    return ret

# def add_input_noise_binary(
#             pattern, prob, f, noise_mask=None, noise_pattern=None):
#     noisy_pattern = copy.copy(pattern)
#     for i in range(len(noisy_pattern)):
#         if random.random() < prob:
#             r = random.random()
#             if r < f:
#                 noisy_pattern[i] = 1
#             else:
#                 noisy_pattern[i] = 0
#     return noisy_pattern
# def add_input_noise_binary(
#             pattern, prob, f=None, noise_mask=None, noise_pattern=None):
#     return np.where(noise_mask, noise_pattern, pattern)

def add_noise_float_patterns(
        patterns, prob, n, seed=None, scaled_noise=False, signal_mask=None):
    if signal_mask:
        assert not scaled_noise
    if seed is not None:
        random.seed(seed)
    out_arr = []
    # pattern_len = len(patterns[0][0])
    # noise_mask = np.zeros(pattern_len, dtype=np.uint8)
    # noise_mask[0:int(prob*pattern_len)] = 1
    # noise_pattern = np.zeros(pattern_len, dtype=np.uint8)
    # noise_pattern[0:int(f*pattern_len)] = 1
    for pattern_output in patterns:
        pattern, output = pattern_output
        for i in range(n):
            new_pattern = add_input_noise_float(pattern, prob, scaled_noise, signal_mask,
                        # noise_mask=noise_mask, noise_pattern=noise_pattern,
                        )
            out_arr.append((new_pattern, output))
    return out_arr

def add_input_noise_float(
        pattern, input_noise, scaled_noise=False, signal_mask=None):
    if input_noise > 0:
        pattern = copy.copy(pattern)
        if scaled_noise:
            p0 = 1-input_noise
            for i in range(len(pattern)):
                r = random.random()
                pattern[i] = pattern[i]*p0 + r*input_noise
        elif signal_mask:
            for i in range(len(pattern)):
                if not signal_mask[i]:
                    if random.random() < input_noise:
                        pattern[i] = random.random()
        else:
            for i in range(len(pattern)):
                if random.random() < input_noise:
                    pattern[i] = random.random()
    return pattern
# def add_input_noise_float(
#             pattern, input_noise, scaled_noise=False, signal_mask=None,
#             scaled_noise=False, signal_mask=None):
#     np.random.shuffle(noise_mask)
#     np.random.shuffle(noise_pattern)
#     return np.where(noise_mask, noise_pattern, pattern)


def add_noise_uniform255_patterns(
        patterns, prob, n, seed=None):
    if seed is not None:
        random.seed(seed)
        rng = np.random.default_rng(seed)
    out_arr = []
    pattern_len = len(patterns[0][0])
    noise_mask = np.zeros(pattern_len, dtype=np.uint8)
    noise_mask[0:int(prob*pattern_len)] = 1
    noise_pattern = rng.integers(low=0, high=256, size=pattern_len).astype(np.uint8)
    for pattern_output in patterns:
        pattern, output = pattern_output
        for i in range(n):
            new_pattern = add_input_noise_binary(pattern, prob,
                            noise_mask=noise_mask, noise_pattern=noise_pattern
                        )
            out_arr.append((new_pattern, output))
    return out_arr

# def add_input_noise_uniform255(
#         pattern, input_noise,
#         noise_mask,
#         noise_pattern,
#         ):
#     if input_noise > 0:
#         pattern = copy.copy(pattern)
#         for i in range(len(pattern)):
#             if random.random() < input_noise:
#                 pattern[i] = random.randint(0, 255)
#     return pattern
# # def add_input_noise_float(
# #             pattern, input_noise, scaled_noise=False, signal_mask=None,
# #             scaled_noise=False, signal_mask=None):
# #     np.random.shuffle(noise_mask)
# #     np.random.shuffle(noise_pattern)
# #     return np.where(noise_mask, noise_pattern, pattern)


