def build_report_mode_annotation(
    *,
    report_mode: str,
    leakage_guard_mode: str,
    smoke_run: bool,
) -> dict | None:
    mode = str(report_mode).strip().lower()
    if mode != "on":
        return None

    leakage_mode = str(leakage_guard_mode).strip().lower()

    if smoke_run:
        return {
            "enabled": True,
            "publishability": "not_publishable",
            "tier": "smoke_speed_only",
            "risk_level": "high",
            "risk_notes": [
                "当前为低轮次冒烟/测速运行，不能作为正式性能结论。",
                "建议切换到epochs>=20的质量评估口径后再用于报告。",
            ],
            "disclosure": "可用于链路可用性与耗时对比，不用于模型优劣结论。",
        }

    if leakage_mode == "strict":
        return {
            "enabled": True,
            "publishability": "publishable_primary",
            "tier": "strict_leakage_guard",
            "risk_level": "low",
            "risk_notes": [
                "字段名泄漏防护最强，显式标签/诊断字段会被大范围过滤。",
                "结果更保守，可能低估数据可达上限。",
            ],
            "disclosure": "适合作为主口径结果；需说明该口径偏保守。",
        }

    if leakage_mode == "balanced":
        return {
            "enabled": True,
            "publishability": "publishable_with_disclosure",
            "tier": "balanced_information_retention",
            "risk_level": "medium",
            "risk_notes": [
                "保留更多高信息临床特征，通常性能更高。",
                "需核验Top特征中是否存在标签代理变量。",
            ],
            "disclosure": "可用于性能主结果，但应补充字段审计与代理泄漏检查说明。",
        }

    return {
        "enabled": True,
        "publishability": "not_publishable",
        "tier": "leakage_guard_off",
        "risk_level": "high",
        "risk_notes": [
            "已关闭字段名泄漏防护，存在不可控标签泄漏风险。",
            "不建议用于可发表/正式对外结论。",
        ],
        "disclosure": "仅建议用于内部探索与误差定位。",
    }
