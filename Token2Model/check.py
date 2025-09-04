def run_checks(all_checks, checklists, skip_results=True):
    for check_item in all_checks:
        check, max_score = check_item
        fail_previous_check = False
        for result in checklists.values():
            if result["score"] != result["max_score"]:
                fail_previous_check = True
                break
        if fail_previous_check and skip_results:
            
            result = {"score": 0, "error_var": [], "message": "Skipped", "max_score": max_score}
            checklists[check.__name__] = result
        else:
            checklists[check.__name__] = check()
    return checklists