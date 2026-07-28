# De-identified human response data

`deidentified_human_stage_trials.csv` contains the human behavioral responses
used in the transfer analyses. Each row is one task stage. Repeated stages from
the same participant share a release-only random `participant_code`.

The public table contains only task assignments, actions, and outcomes. The
following source fields were removed before release:

- original participant/platform identifier;
- session UUID;
- internal assignment identifier;
- all timestamps and local file paths.

No names, email addresses, IP addresses, free-text responses, ages, genders, or
other demographic fields are present. The private mapping from original
participant identifiers to release codes was not written to disk and cannot be
recovered from this repository.

## Columns

- `participant_code`: random release-only participant alias.
- `pair_condition`: relation between the two task stages.
- `unordered_pair_index`, `human_unordered_pair_id`: task-pair identifiers.
- `condition`: upward or downward tool mechanism.
- `position`: first or second stage within the assigned pair.
- `family`, `env_id`, `variant`: task configuration.
- `solved`, `attempts`: terminal task outcome and attempts used.
- `first_attempt_success`, `first_attempt_x`, `first_attempt_y`: first action.
- `success_attempt`, `success_x`, `success_y`: successful action when solved;
  blank when unsolved.

## Privacy audit

`privacy_audit.json` records the released row/participant counts, excluded
identifier classes, and the SHA-256 digest of the exact CSV. The release was
also scanned for direct identifiers, UUIDs, email addresses, IP addresses,
absolute local paths, and demographic column names.
