"""Pure summary calculations; policy claims and offline measurements stay separate."""

import json

def diagnostic_summary(rows):
    reports = []
    for row in rows:
        reported = list(row.get("policy_reports", []))
        current = row.get("policy_current_report")
        if isinstance(current, dict):
            encode = lambda item: json.dumps(item, sort_keys=True, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
            if not any(report is current or encode(report) == encode(current) for report in reported):
                reported.append(current)
        reports.extend(report for report in reported if not report.get("skipped"))
    audited = [item for row in rows for item in row.get("offline_batch_audit", []) if item.get("attempted")]

    def rate(items, field):
        known = [item for item in items if item.get(field) is not None]
        passed = sum(bool(item[field]) for item in known)
        return {"passed": passed, "observed": len(known), "unknown": len(items) - len(known),
                "rate": passed / len(known) if known else None}

    conditional = []
    conditional_unknown = 0
    for row in rows:
        batch = sorted(row.get("offline_batch_audit", []), key=lambda item: item["batch_index"])
        previous = True
        seen_previous = False
        previous_items = []
        for item in batch:
            if seen_previous and previous and item.get("attempted"):
                start = item.get("first_attempt_step")
                times = [prior.get("first_valid_placement_step") for prior in previous_items]
                if start is None or any(time is None for time in times):
                    conditional_unknown += 1
                elif all(time <= start for time in times):
                    conditional.append(item)
            previous = previous and item.get("valid_placement") is True
            seen_previous = True
            previous_items.append(item)
    recoveries = [event for row in rows for event in row.get("policy_recovery_events", [])]
    recovery_known = [event for event in recoveries if event.get("success") is not None or event.get("recovered") is not None]
    recovery_passed = sum(bool(event.get("success", event.get("recovered"))) for event in recovery_known)
    done = [row for row in rows if row.get("phase") == "DONE" or
            any(tr.get("to") == "DONE" for tr in row.get("policy_transitions", []))]
    return {
        "offline_truth_audit_episodes": sum(bool(row.get("offline_batch_audit")) for row in rows),
        "offline_truth_planned_nonzero_batches": sum(len(row.get("offline_batch_audit", [])) for row in rows),
        "offline_truth_attempted_batches": len(audited),
        "offline_truth_grasp": rate(audited, "valid_grasp"),
        "offline_truth_placement": rate(audited, "valid_placement"),
        "offline_truth_final_retention": rate(audited, "final_valid_placement"),
        "subsequent_batch_placement_given_previous_valid": rate(conditional, "valid_placement"),
        "subsequent_batch_condition_timing_unknown": conditional_unknown,
        "controller_reported_grasp_count": rate(reports, "count_verified"),
        "controller_reported_placement": rate(reports, "placement_verified"),
        "recovery_events": len(recoveries),
        "explicitly_labeled_recovery_successes": recovery_passed,
        "explicitly_labeled_recovery_observations": len(recovery_known),
        "recovery_success_rate": recovery_passed / len(recovery_known) if recovery_known else None,
        "done_episodes": len(done), "done_but_official_failure": sum(not row["success"] for row in done),
        "note": "Offline audit samples at stage transitions and every five controls. Grasp requires exactly the intended elevated cookies connected to the robot; placement requires target geometry, seating, settled speed and release at an observed sample. This diagnostic rate does not replace the official episode score. Controller claims are separate.",
    }
