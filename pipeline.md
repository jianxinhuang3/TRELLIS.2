我希望实现从rgb卫星图生成基于mesh的3d城市。现在我希望先基于TRELLIS.2实现楼房的生成。
你可以拿到的条件是：
- 屋顶顶视图作为外观A（在真实卫星图中，可以用sam3做实例分割）
- 挤出的3d白模作为几何G（在真实卫星图中，可以用分割得到footprint + 深度估计建筑高度 来挤出白模）
- 周围街区/建筑的图片作为上下文风格C（在test_on_train阶段，先跳过此步骤）


═══════════════ PART T0: 离线数据构造（SatSkylines 44k，只跑一次）═══════════════

每个GT资产 asset.glb（+ labels.tsv: height/size/item_class）
│
├── 目标侧（frozen encoder构造flow matching监督目标）
│   ├── 表面体素化64³ → ss_enc❄️ → z*_ss [8,16,16,16]
│   ├── dual-grid体素化 → shape_enc❄️ → (z−μ)/σ
│   │      → z*_shape [Mg,32] + GT粗coords Sg [Mg,4]
│   └── PBR体素化 → tex_enc🆕❄️ → (z−μt)/σt
│          → z*_tex [Mg,32]
│
├── 条件侧（刻意退化，模拟真实推理输入）
│   ├── 外观A: 顶视渲染 → 卫星风格增广 → roof.png
│   ├── 几何G: GT轮廓简化/扰动 + 高度误差 → 挤出proxy
│   │      → proxy_voxel
│   │      → 训练时 ss_enc❄️ → GeoTokenizer🔥 → G tokens
│   └── 上下文风格C: 伪街区分组 → K个邻居顶视渲染 + 相对位姿 → C tokens
│
└── 落盘:
    {z*_ss, z*_shape, Sg, z*_tex,
     roof.png, proxy_voxel, neighbor_ids, relative_poses}

注：z*_shape和z*_tex必须使用相同Sg并按coords严格对齐。


═══════════════ PART T1: SS flow 注入层训练 ════════════════

t ~ U(0,1), ε ~ N(0,I)
x_t = (1−t)·ε + t·z*_ss                     [B,8,16,16,16]
│
├── 条件: {A,G,C}，独立CFG drop
└── ss_flow_img_dit❄️ + 解耦注入🔥 + gate🔥
    └── v̂_ss
        └── Loss_SS = ‖v̂_ss − (z*_ss−ε)‖²

训练只做单步前向，仅更新🔥参数。

推理：
ε → SS flow多步采样 → ẑ_ss → ss_dec❄️ → predicted support Ŝ

注：从纯噪声开始；proxy只作为G条件，不使用SDEdit初始化。


═══════════════ PART T2: Shape flow 注入层训练 ════════════════

基础训练：
coords = Sg                                  # 不运行Stage 1
t ~ U(0,1), ε ~ N(0,I)
x_t = (1−t)·ε + t·z*_shape                  [Mg,32]
│
├── 条件: {A,G,C}，独立CFG drop
└── slat_flow_img2shape_512❄️ + 解耦注入🔥 + gate🔥
    └── v̂_shape
        └── Loss_shape = ‖v̂_shape−(z*_shape−ε)‖²

鲁棒性训练：
逐步混入扰动support或Stage 1预测support Ŝ，
降低训练使用Sg、推理使用Ŝ造成的差异。

前提：更换support时，必须使用该support上正确对齐的shape目标；
若无法获得对应目标，则保持Sg训练，仅用Ŝ做完整管线验证。


═══════════════ PART T3: Tex flow 注入层训练 ════════════════

coords = Sg
t ~ U(0,1), ε ~ N(0,I)
x_t = (1−t)·ε + t·z*_tex                    [Mg,32]
│
├── concat_cond:
│   初始使用 z*_shape
│   后续逐步混入 noisy/generated shape latent
├── 条件: {A,C}，独立CFG drop
└── slat_flow_imgshape2tex_512❄️ + 解耦注入🔥 + gate🔥
    └── v̂_tex
        └── Loss_tex = ‖v̂_tex−(z*_tex−ε)‖²

先固定Sg，只处理GT shape → generated shape的特征差异。

若能在预测support Ŝ上获得正确对齐的texture目标，
再扩展为：
coords=Ŝ，concat_cond=ẑ_shape,Ŝ，target=z*_tex,Ŝ。


═══════════════ PART T4: 动态预测缓存（训练后期）═══════════════

Stage 1 EMA完整采样 → Ŝ
Stage 2 EMA完整采样 → ẑ_shape

用于：
├── Shape的predicted-support鲁棒性训练
└── Texture的generated-shape鲁棒性训练

这些缓存依赖当前模型，因此应周期性刷新，不属于一次性T0。


═══════════════ PART T5: 训练配置与验证 ════════════════

optimizer: AdamW，仅🔥参数(<80M) | bf16 | EMA | 16×H20
│
└── 验证:
    ├── 几何: footprint/support IoU、 高度误差
    ├── Shape: Sg上的Oracle结果 vs Ŝ上的端到端结果
    ├── 外观: 生成渲染 vs GT渲染的DINO/CLIP相似度
    ├── 风格: 同伪街区两两DINO相似度及C分支消融
    └── 基座无损: gate=0时与原版TRELLIS.2一致（只是实现正确性检查，不是持续训练目标，也不需要加入 loss）
