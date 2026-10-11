# N-BaIoT per-device class coverage (pre-E2 check)

**Source:** E2 `scale_on=train` cache `nbaiot_4ddb8b93b8821650.npz` **train** split (`n=1,418,112`), 20,000-samples-per-file cap, `GLOBAL_SEED=42` stratified label split. Re-counted 2026-09-21 after `--build-cache-only`. Train client sizes and missing-class pattern match the earlier full-scaler recount (`a9dd0c758635aebe`); that is expected (same split, labels and device ids unchanged). Missing-class pattern is a **dataset** property (Ennio and Samsung webcam have no Mirai files).

Client index = `LabelEncoder` order of device names:

| k | Device | Train n | Classes present | Missing |
|---|--------|--------:|----------------:|---------|
| 0 | danmini_doorbell | 175931 | 11/11 | — |
| 1 | ecobee_thermostat | 170558 | 11/11 | — |
| 2 | ennio_doorbell | 95875 | **6/11** | all five `mirai_*` |
| 3 | philips_b120n10_baby_monitor | 176014 | 11/11 | — |
| 4 | provision_pt_737e_security_camera | 176083 | 11/11 | — |
| 5 | provision_pt_838_security_camera | 175901 | 11/11 | — |
| 6 | samsung_snh_1011_n_webcam | 96007 | **6/11** | all five `mirai_*` |
| 7 | simplehome_xcs7_1002_wht_security_camera | 176174 | 11/11 | — |
| 8 | simplehome_xcs7_1003_wht_security_camera | 175569 | 11/11 | — |

**Implication:** a device-based FL partition is **not** 9 copies of the same 11-class task. Two clients never see Mirai in training. Macro-F1 on the **pooled** test set can therefore move because of **missing classes on some clients**, not only because of aggregation. E2 must report this; it must **not** be described as nine equivalent IoT deployments.

**Scaler (E2):** `scale_on=train` fits `StandardScaler` on the **pooled training matrix** (all devices together), then transforms train and test. That is **not** fully local / privacy-preserving preprocessing (no per-device scaler; test still uses the pooled-train scaler).

**Test-device IDs:** E2 caches **must** store `cid_test`. After cache build, require all **9** device ids to appear in the pooled test split (otherwise the “pooled holdout, not unseen-device” claim cannot be audited).
