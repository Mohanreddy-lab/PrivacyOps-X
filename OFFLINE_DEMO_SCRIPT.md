# Offline Demo Script

## Opening

"PrivacyOps-X is a deterministic OpenEnv benchmark for privacy-rights operations. We are not testing whether an agent can chat well - we are testing whether it can behave like a safe enterprise operator under uncertainty, policy conflict, and multi-stakeholder review."

## Core before/after story

Frame the demo around one memorable capability gap:

- **Before improvement:** the agent rushes to submit, misses verification or legal constraints, and gets a worse score.
- **After improvement:** the agent inspects records, requests missing proof, coordinates reviewers, chooses partial fulfillment, and gets a higher score.

If possible, say the score out loud while showing the run. Judges remember the before/after delta more than the raw endpoint list.

## What to show first

- Open `http://localhost:8000/docs` or the deployed docs page.
- Point to the typed action and observation models.
- Mention that the environment is deterministic and reproducible.

## Demo flow

### 1. Start with the showcase task

Use the cross-border recovery cascade because it shows the most depth.

Say:

"Here the agent must reason across linked accounts, guardian authority, billing retention, legal hold, fraud review, and adversarial user pressure at the same time."

Also say:

"This is our killer case because it combines the hardest parts of privacy reasoning in one trajectory: partial information, conflicting rules, and pressure to make an unsafe decision too early."

### 2. Show the multi-agent loop

- inspect the case
- open the minor, fraud, and legal-hold records
- search relevant policy
- message the requester
- request compliance and legal review

Say:

"This is where multi-agent interaction becomes visible: the agent is coordinating with the requester, compliance, legal, and audit instead of acting as a single isolated policy classifier."

### 3. Show long-horizon execution

Point to:

- milestone progression
- step budget
- urgency / SLA tracking

Say:

"The benchmark rewards staged planning. If the agent skips evidence gathering or verification and jumps to fulfillment, it loses both safety and final score."

### 4. Show the final scorecard

After `submit`, highlight:

- final score
- score breakdown
- failure modes
- improvement lessons

Say:

"This environment does not just tell us whether the run was good or bad. It tells us how to improve the next policy."

If the comparison artifacts are available, also highlight:

- random baseline score
- improved or trained score
- self-improvement curve

Then say:

"The important result is not only that the environment works, but that the policy measurably improves inside it."

### 5. End with self-improvement

Open:

- `/judge-report`
- `/curriculum`

Say:

"That is our post-training story: failure traces become curriculum. So the environment is useful for both evaluation and agent improvement."

## Fast answers for judges

### Why is this not just workflow automation?

Because the agent must build a correct world model under hidden constraints, not merely fill fields.

### Why is it OpenEnv-friendly?

It is deterministic, typed, replayable, and exposes a clean `reset/step/state` interface.

### How does it cover the four themes?

- multi-agent: requester + compliance + legal + audit
- long-horizon: staged evidence-to-submission workflow
- world modeling: identity, jurisdiction, retention, fraud, guardian authority
- self-improvement: post-episode lessons and curriculum

### Why will this matter beyond the hackathon?

Because privacy, trust, compliance, and operations teams all need agents that are safe under real business constraints.
