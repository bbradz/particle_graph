import grammar.lie_algebra as lie

def _make_anomaly_checklist(gauge_groups: dict):
    """
    Make a checklist for gauge anomaly checks.
    """
    abelian_groups = [g_id for g_id, group in gauge_groups.items() if lie.is_abelian(group["type"], group["rank"])]
    non_abelian_groups = [g_id for g_id, group in gauge_groups.items() if not lie.is_abelian(group["type"], group["rank"])]

    anomaly_checklist = {"G^3":[], "U1-G^2":[], 'U1^3':[], 'U1-grav':[],"U1-mixing":[]}

    # (1) Non-abelian cubic anomalies (AAA)
    for na_group in non_abelian_groups:
        anomaly_checklist["G^3"].append([na_group, na_group, na_group])
        
        # (2) Mixed non-abelian^2 - abelian anomalies (AAB)
        for a_group in abelian_groups:
            anomaly_checklist["U1-G^2"].append([na_group, na_group, a_group])

    # (3) Pure abelian cubic anomalies (BBB)
    for a_group in abelian_groups:
        anomaly_checklist["U1^3"].append([a_group, a_group, a_group])

        # (4) Gravitational - abelian anomalies (GAA)
        anomaly_checklist["U1-grav"].append([a_group, a_group, "grav"])

    # (5) Abelian group mixing anomalies (BBC, BCD)
    for i in range(len(abelian_groups)):
        for j in range(len(abelian_groups)):
            if i != j:
                anomaly_checklist["U1-mixing"].append([abelian_groups[i], abelian_groups[i], abelian_groups[j]])
        for j in range(i+1, len(abelian_groups)):
            for k in range(j+1, len(abelian_groups)):
                anomaly_checklist["U1-mixing"].append([abelian_groups[i], abelian_groups[j], abelian_groups[k]])
    return anomaly_checklist

def non_abelian_cubic_anomaly(group1, rep1, chirality, dim):
    chirality = 1 if chirality == "left" else -1
    group_type = group1["type"].split("_")[1]
    N = int(group1["rank"].split("_")[1])

    anomaly_coefficient = lie.SU_N_cubic_anomaly(N, rep1)
    return anomaly_coefficient * chirality * dim


if __name__ == "__main__":
    gauge_groups = {'g_1': {'id': 'g_1', 'type': 'GAUGE_U', 'rank': 'rank_1'}, 
                    'g_2': {'id': 'g_2', 'type': 'GAUGE_SU', 'rank': 'rank_2'}, 
                    'g_3': {'id': 'g_3', 'type': 'GAUGE_SU', 'rank': 'rank_3'}}

    result = non_abelian_cubic_anomaly(gauge_groups["g_2"], 'fnd', 'left', 1)
    print(result)
    