conf_patchtst_standard = {
    'modelname': 'PatchTST_standard',
    'modeltype': 'PATCHTST',
    'parameters': dict(
        patch_len=16,
        stride=8,
        n_layers=3,
        d_model=128,
        n_heads=16,
        d_ff=256,
        dropout=0.2,
        lr=1e-3,
        epochs=30,
        batch_size=64,
        revin=False
    )
}
