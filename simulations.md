# Run commands to replicate results

Collection of commands that call python scripts to replicate the results from the paper.
To reproduce the figures in the paper, run the simulation command below to generate the raw data, then run the corresponding Figure Notebook in the `notebooks` folder.


## Figure 2 & 3

Runs directly in the notebook, see `notebooks/fig2_fig3.ipynb`.

---

## Figure 4: Trajectory learning

#### BCP & Control runs (5 networks each)

```bash
python run_traj.py -m +experiment=fig4_bcp    # BCP
python run_traj.py -m +experiment=fig4_nofb   # No feedback control
python run_traj.py -m +experiment=fig4_nohl   # No hidden layer control
```
**Note:** `fig4_bcp` is also needed as a baseline reference for several supplementary figures.

#### Example networks with recorded activity:

```bash
python run_traj.py +experiment=fig4_example
python run_traj.py +experiment=fig4_example_nofb
python run_traj.py +experiment=fig4_example_nohl
```

---

## Figure 5: Trajectory learning (overlapping assemblies)

#### Sweep over different overlap levels (5 networks each)

```bash
python run_traj.py -m +experiment=fig5_overlap
```

#### One example network per overlap level with recorded activity

```bash
python run_traj.py -m +experiment=fig5_overlap_examples
```

**Note:** Panel g also needs the Figure 4 no-hidden-layer runs (`out/fig4_nohl`).

---

## Figure 6: Fashion-MNIST

#### BCP networks
**Note:** These should be run on GPU (add `device=gpu` to the run commands below. Training can take several hours per network, depending on your system.)

```bash
python run_static.py -m +experiment=fig6_fmnist_1l   # 1 hidden layer, overlap 0-10%, 6 x 5 seeds (panels b-j)
python run_static.py -m +experiment=fig6_fmnist_3l   # 3 hidden layers, 5 seeds (b-d)
```

#### Controls 

These are all fully connected networks trained with backprop, can also run on CPU rather quickly with `device=cpu`.

```bash
python run_static.py -m +experiment=fig6_fmnist_bp_1l            # backprop, 1 hidden layer (d, g)
python run_static.py -m +experiment=fig6_fmnist_bp_3l            # backprop, 3 hidden layers (d)
python run_static.py -m +experiment=fig6_fmnist_fixedhidden_1l   # only the readout trained, 1 hidden layer (d)
python run_static.py -m +experiment=fig6_fmnist_fixedhidden_3l   # only the readout trained, 3 hidden layers (d)
python run_static.py -m +experiment=fig6_fmnist_nohl             # no hidden layer (g; also Figure 7 q)
```

**Note:** `fig6_fmnist_3l` and `fig6_fmnist_bp_3l` are also the analytic-feedback and backprop references of Figure 7.

---


## Figure 7: Online learning of feedback weights

#### Student-teacher task 

```bash
python run_static.py -m +experiment=fig7_st_learned     # learned feedback (panels b-c, e-i)
python run_static.py -m +experiment=fig7_st_random      # fixed random feedback (f-i, m)
python run_static.py -m +experiment=fig7_st_analytic    # analytic feedback (g)
python run_static.py -m +experiment=fig7_st_frequency   # offline-phase frequency sweep, 5 x 3 seeds (k-m)
```

#### Fashion-MNIST (use GPU! Takes ~9-12 h per network)

```bash
python run_static.py -m +experiment=fig7_fmnist_learned      # learned feedback
python run_static.py -m +experiment=fig7_fmnist_pretrained   # pretrained, then frozen feedback
python run_static.py -m +experiment=fig7_fmnist_random       # fixed random feedback
```

**Note:** Panels n, o, q also need the Figure 6 runs `fig6_fmnist_3l`, `fig6_fmnist_bp_3l` and `fig6_fmnist_nohl`
as analytic-feedback, backprop and no-hidden-layer references. Use `gpu_id=N` to pick the GPU.

---

## Figure 8: Fear conditioning task

```bash
python run_fearcond.py -m +experiment=fig8_fearcond_control
python run_fearcond.py -m +experiment=fig8_fearcond_archt
```

---

## Figure 9: Motor learning task

```bash
python run_motor.py -m +experiment=fig9_motor
```

---

## Figure 10: Experimental predictions

Runs directly in the notebook, see `notebooks/fig10.ipynb`.

---

## Supplementary Figure 2: Random feedback

#### 5 networks per random feedback condition.
```bash
python run_traj.py -m +experiment=supp_fig2_randfb_assembly
python run_traj.py -m +experiment=supp_fig2_randfb_neuron
```

**Note:** Figure requires `fig4_bcp` output as a reference too

---

## Supplementary Figure 3: E-E and I-I connections

```bash
python run_traj.py -m +experiment=supp_fig3_ee_ii
```

---

## Supplementary Figure 4: Feedback to excitatory neurons

```bash
python run_traj.py -m +experiment=supp_fig4_fb_to_e
```

---

## Supplementary Figure 5: E-PV-SOM-VIP circuit

#### Panels b-h
```bash
python run_traj.py -m +experiment=supp_fig5_pv_som_vip
```

#### Panel i (connection parameter sweeps)
```bash
python run_traj.py -m +experiment=supp_fig5_sweep_g_sompv
python run_traj.py -m +experiment=supp_fig5_sweep_g_somvip
python run_traj.py -m +experiment=supp_fig5_sweep_g_evip
python run_traj.py -m +experiment=supp_fig5_sweep_g_epv
```

---

## Supplementary Figure 6: Learning I-to-E connections

```bash
python run_traj.py -m +experiment=supp_fig6_learning_ie
```
