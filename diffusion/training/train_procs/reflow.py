
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from hydra.utils import instantiate
from diffusion.training.loop import fit_one
from diffusion.training.load_save_ckpt import save_checkpoint, load_model
from diffusion.diffusionprocess import DiffusionProcess
from diffusion.solvers.euler import EulerSolver
from diffusion.training.train_procs.standard import StandardProc


class ReflowProc:
    def __init__(self, diffusion_process, **kw):
        self.dp = diffusion_process

    def training_loss(self, batch):
        x0, x1 = batch
        return self.dp.training_loss(x0, noise=x1)


def make_reflow_pairs(cfg, teacher, pmtrz, tp, sched, x0_data, device):
    """
    Reflow pairs are made by integrating the PF-ODE bacwards
    """
    N = int(cfg.solver.N)
    #The other solvers are more diffiuclt to integrate backwards
    solver = EulerSolver(model=teacher, parameterization=pmtrz,
                         time_param=tp, N=N, schedule=sched)
    #Flip time grid to reverse the PF Ode
    rev = torch.flip(tp.grid(N, device=device), dims=[0])
    bs = cfg.trainer.batch_size
    x1_chunks = []
    for i in range(0, len(x0_data), bs):
        xb = x0_data[i:i + bs].to(device)
        x1 = solver.sample(xb, time_grid=rev)
        x1_chunks.append(x1.detach().cpu())
    x1 = torch.cat(x1_chunks, dim=0)
    return TensorDataset(x0_data.cpu(), x1)


def run_reflow(cfg, sched, tp, t_distribution, weight_func, pmtrz,
               train_set, val_set, device, save_dir):

    x0_train = train_set
    x0_val = val_set

    teacher_path = cfg.train_proc.get("teacher_path", None)
    if teacher_path:
        #we can load the teacher
        teacher, _, _, _ = load_model(teacher_path, sched, device=device)
        #TODO check that the loaded parameterizations are not different than the live param?
    else:
        teacher = instantiate(cfg.model).to(device)
        DP0 = DiffusionProcess(schedule=sched, parameterization=pmtrz,
                               weight_func=weight_func, network=teacher,
                               t_distribution=t_distribution, time_param=tp)
        std = StandardProc(DP0)
        tl = DataLoader(train_set, batch_size=cfg.trainer.batch_size, shuffle=True)
        vl = DataLoader(val_set, batch_size=cfg.trainer.batch_size, shuffle=False)
        print("[reflow] training standard teacher")
        teacher, opt0, _, _, _ = fit_one(teacher, std, tl, vl, cfg, device)
        save_checkpoint(os.path.join(save_dir, "reflow_0.pt"),
                        cfg.trainer.epochs, teacher, opt0, cfg)
        
    for k in range(1, cfg.train_proc.n_steps + 1):
        print(f"[reflow] round {k}: generating pairs from teacher")
        train_pairs = make_reflow_pairs(cfg, teacher, pmtrz, tp, sched, x0_train, device)
        val_pairs = make_reflow_pairs(cfg, teacher, pmtrz, tp, sched, x0_val, device)
        tl = DataLoader(train_pairs, batch_size=cfg.trainer.batch_size, shuffle=True)
        vl = DataLoader(val_pairs, batch_size=cfg.trainer.batch_size, shuffle=False)

        student = instantiate(cfg.model).to(device)
        DP = DiffusionProcess(schedule=sched, parameterization=pmtrz,
                              weight_func=weight_func, network=student,
                              t_distribution=t_distribution, time_param=tp)
        trainproc = ReflowProc(DP)
        print(f"[reflow] round {k}: training student")
        student, opt, _, _, _ = fit_one(student, trainproc, tl, vl, cfg, device)
        save_checkpoint(os.path.join(save_dir, f"reflow_{k}.pt"),
                        cfg.trainer.epochs, student, opt, cfg)
        teacher = student   

    return teacher
