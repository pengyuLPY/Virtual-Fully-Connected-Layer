def adjust_learning_rate(lr_obj, optimizer, Iteration):
    lr_policy = lr_obj.lr_policy
    base_lr = lr_obj.base_lr
    if lr_policy == 'multistep':
        steps = lr_obj.steps
        gamma = lr_obj.gamma
        multistep(base_lr, gamma, optimizer, Iteration, steps)

def multistep(base_lr, gamma, optimizer, Iteration, steps=[]):
    current_step = 0
    for step in steps:
        if Iteration >= step:
            current_step += 1
    lr = pow(gamma,current_step) * base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
