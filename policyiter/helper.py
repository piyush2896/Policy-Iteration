from collections import defaultdict
from copy import deepcopy

def eval_polciy(policy, world, gamma=0.99, threshold=1e-10, noise=0.8):
    V_vals = defaultdict(lambda: 0.)
    for state in world.pos_reward_states:
        V_vals[state] = world.pos_reward_vals[state]
    for state in world.neg_reward_states:
        V_vals[state] = world.neg_reward_vals[state]

    while True:
        for state in world.states:
            v_state = 0.
            prev_v = deepcopy(V_vals)
            for action in policy[state]: # policy is a dictionary of state. policy[state] is a dictionary of actions
                if state in world.pos_reward_states:
                    v_state = world.pos_reward_vals[state]
                    continue
                if state in world.neg_reward_states:
                    v_state = world.neg_reward_vals[state]
                    continue

                _, n_state = world.move_given_action(state, action)
                v_n =  ((1 - noise) * (world.get_reward(state, action, n_state) +
                    V_vals[n_state] * gamma))
                lr_states = world.move_lr_given_action(state, action)
                v_l =  (noise / 2 * (world.get_reward(state, action, lr_states[0]) +
                    V_vals[lr_states[0]] * gamma))
                v_r =  (noise / 2 * (world.get_reward(state, action, lr_states[1]) +
                    V_vals[lr_states[1]] * gamma))

                v_state += policy[state][action] * (v_n + v_l + v_r)
            V_vals[state] = v_state
        if sum([abs(prev_v[key] - V_vals[key]) for key in V_vals]) <= threshold:
            #import pdb; pdb.set_trace()
            break
    return dict(V_vals)

def init_policy(world):
    policy = defaultdict(lambda: {})
    for state in world.states:
        actions = world.actions_available(state)
        for action in actions:
            policy[state][action] = 1. / len(actions)
    return policy

def one_step_lookahead(world, state, V, noise, gamma):
    if state in world.pos_reward_states or state in world.neg_reward_states:
        return world.EXIT
    v_a_pairs = []
    actions = world.actions_available(state)
    for action in actions:
        _, n_state = world.move_given_action(state, action)
        v_n = ((1 - noise) * (world.get_reward(state, action, n_state) +
            V[n_state] * gamma))
        lr_states = world.move_lr_given_action(state, action)
        v_l = (noise / 2 * (world.get_reward(state, action, lr_states[0]) +
            V[lr_states[0]] * gamma))
        v_r = (noise / 2 * (world.get_reward(state, action, lr_states[1]) +
            V[lr_states[1]] * gamma))
        v_a_pairs.append((v_n + v_l + v_r, action))
    return max(v_a_pairs)[1]

def one_hot_policy_state(policy, state, action):
    for a in policy[state]:
        policy[state][a] = 1. if a == action else 0.
    return policy

def finalize_pi(policy):
    optim_policy = {}
    for state in policy:
        optim_policy[state] = max(list(zip(policy[state].values(), policy[state].keys())))[1]
    return optim_policy

def improve_policy(world,
                   gamma=0.99,
                   threshold=1e-5,
                   noise=0.8,
                   policy_init=None,
                   verbose=10,
                   max_iter=1000):
    if policy_init is None:
        policy = init_policy(world)
    else:
        policy = policy_init

    v_ctr = 0

    while True:
        #import pdb; pdb.set_trace()
        V = eval_polciy(policy, world, gamma, threshold, noise)
        is_policy_stable = True

        for state in world.states:
            choosen_action = max(list(zip(policy[state].values(), policy[state].keys())))[1]
            best_action = one_step_lookahead(world, state, V, noise, gamma)
            if choosen_action != best_action:
                is_policy_stable = False
            policy = one_hot_policy_state(policy, state, best_action)
        if v_ctr % verbose == 0:
            world.display_world_v_vals(V)
            #import pdb; pdb.set_trace()
        v_ctr += 1
        if is_policy_stable or v_ctr == max_iter:
            return finalize_pi(policy), V
