def run_checks(all_checks, checklists, skip_results=True):
    for check in all_checks:
        fail_previous_check = False
        for result in checklists.values():
            if result["score"] != result["max_score"]:
                fail_previous_check = True
                break
        if fail_previous_check and skip_results:
            # Get the max_score from the check function to maintain consistency
            temp_result = check()
            max_score = temp_result.get("max_score", 0)
            result = {"score": 0, "error_var": [], "message": "Skipped due to previous check failure", "max_score": max_score}
            checklists[check.__name__] = result
        else:
            checklists[check.__name__] = check()
    return checklists