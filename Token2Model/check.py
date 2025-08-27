def run_checks(all_checks, checklists, skip_results=True):
    for check in all_checks:
        fail_previous_check = False
        for result in checklists.values():
            if result["score"] != 1:
                fail_previous_check = True
                break
        if fail_previous_check and skip_results:
            result = {"score": 0, "error_var": [], "message": "Skipped due to previous check failure"}
            checklists[check.__name__] = result
        else:
            checklists[check.__name__] = check()
    return checklists