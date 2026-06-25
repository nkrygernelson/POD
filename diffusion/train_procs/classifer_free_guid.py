def run_reflow(cfg, sched, tp, t_distribution, weight_func, pmtrz,
               val_loader, device, save_dir):
    # round 0 teacher: load a pretrained standard model, or train one first.
    # TODO: load teacher from cfg.train_proc.teacher_path
    teacher = ...  # network

    for k in range(cfg.train_proc.n_steps):
        # 1. build a solver on the TEACHER and sample x1 -> x0 pairs
        # TODO: solver = instantiate(cfg.solver, model=teacher, time_param=tp,
        #                            schedule=sched, parameterization=pmtrz, N=cfg.solver.N)
        # TODO: pairs = make_reflow_pairs(solver, prior, n_pairs)   # -> Dataset of (x1, x0)
        pairs = ...
        loader = DataLoader(pairs, batch_size=cfg.trainer.batch_size, shuffle=True)

        # 2. fresh student, fresh DP, reflow loss
        student = instantiate(cfg.model)
        DP = DiffusionProcess(schedule=sched, parameterization=pmtrz,
                              weight_func=weight_func, network=student,
                              t_distribution=t_distribution, time_param=tp)
        trainproc = instantiate(cfg.train_proc, diffusion_process=DP, _recursive_=False)

        # 3. SAME loop as standard
        student, opt, _, _ = fit_one(student, trainproc, loader, val_loader, cfg, device)

        save_checkpoint(os.path.join(save_dir, f"reflow_{k}.pt"), cfg.trainer.epochs,
                        student, opt, cfg)
        teacher = student   # next round bootstraps on this

    return teacher