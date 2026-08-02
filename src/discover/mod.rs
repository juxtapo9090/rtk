//! Scans AI coding sessions to find commands that could benefit from RTK filtering.

pub mod lexer;
pub mod provider;
pub mod registry;
mod report;
pub mod rules;

use anyhow::Result;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

use provider::{ClaudeProvider, SessionProvider};
use registry::{
    category_avg_tokens, classify_command, split_command_chain, strip_disabled_prefix,
    Classification,
};
use report::{DiscoverReport, SupportedEntry, UnsupportedEntry};

use crate::core::tracking::{HookDecisionRecord, Tracker};
use crate::discover::registry::prefix_contains_rtk_disabled;
use crate::hooks::hook_check::{status as hook_status, HookStatus};
use crate::hooks::permissions::{self, PermissionVerdict};

/// Whether a `Supported` command was actually routed through RTK, and how
/// confident we are in that answer.
///
/// `Measured` comes from a real `hook_decisions` log row, keyed by the
/// transcript's `tool_use_id` — ground truth about what the hook actually did
/// at the time. `Estimated` is the fallback for history that predates hook-decision
/// logging (or isn't Claude Code): a best-effort guess using *today's*
/// hook-install state, config, and registry, which may not reflect reality at
/// the time the command actually ran.
enum Coverage {
    Measured(bool),
    Estimated(bool),
}

impl Coverage {
    fn is_covered(&self) -> bool {
        match self {
            Coverage::Measured(covered) | Coverage::Estimated(covered) => *covered,
        }
    }

    fn is_estimated(&self) -> bool {
        matches!(self, Coverage::Estimated(_))
    }
}

/// Determine whether `cmd` was (or, absent a log entry, likely would have been)
/// routed through RTK by the installed PreToolUse hook.
///
/// Known imprecision: the caller splits a chained command (`a && b`) into parts
/// and calls this once per part with the *same* `tool_use_id`, since Claude Code's
/// PreToolUse hook fires once per tool call for the whole raw command. A `Deny`/
/// `Ask` verdict genuinely is chain-wide (`permissions::check_command_with_rules`
/// denies/asks the entire chain if any segment matches), but the real rewrite
/// (`registry::rewrite_command`) is best-effort *per segment* — it can rewrite
/// just `a` and leave an unsupported `b` untouched. Neither `Measured` (one
/// `hook_decisions` row per `tool_use_id`, no per-segment detail) nor `Estimated`
/// re-derives that per-segment split, so a segment can be reported covered/missed
/// based on a sibling segment's fate. Not worth chasing further: `Estimated` is a
/// transitional fallback for history that predates `hook_decisions` logging and
/// naturally fades out as that log backfills — after ~90 days only `Measured`
/// should remain in practice.
fn hook_coverage(
    cmd: &str,
    tool_use_id: &str,
    hook_log: &HashMap<String, HookDecisionRecord>,
    hook_installed: bool,
    excluded: &[String],
    transparent_prefixes: &[String],
) -> Coverage {
    if let Some(record) = hook_log.get(tool_use_id) {
        return Coverage::Measured(record.decision.is_covered());
    }
    Coverage::Estimated(estimate_hook_coverage(
        cmd,
        hook_installed,
        excluded,
        transparent_prefixes,
    ))
}

/// Best-effort guess at whether the *currently* installed hook would rewrite `cmd`,
/// used only when no `hook_decisions` log row exists for this exact invocation.
/// Reads real permission config — see `estimate_hook_coverage_with_verdict` for the
/// pure decision core.
fn estimate_hook_coverage(
    cmd: &str,
    hook_installed: bool,
    excluded: &[String],
    transparent_prefixes: &[String],
) -> bool {
    let verdict = permissions::check_command_for(cmd, permissions::Host::Claude);
    estimate_hook_coverage_with_verdict(
        cmd,
        verdict,
        hook_installed,
        excluded,
        transparent_prefixes,
    )
}

/// Pure core of `estimate_hook_coverage`, taking the permission verdict directly so
/// it's testable without depending on real config files. Mirrors the real hook's own
/// decision order (`hooks::hook_cmd::decide_from_verdict`): a `Deny`-rule match or an
/// unattestable construct means the hook would never have rewritten this, regardless
/// of registry support.
fn estimate_hook_coverage_with_verdict(
    cmd: &str,
    verdict: PermissionVerdict,
    hook_installed: bool,
    excluded: &[String],
    transparent_prefixes: &[String],
) -> bool {
    hook_installed
        && verdict != PermissionVerdict::Deny
        && !lexer::contains_unattestable_construct(cmd)
        && registry::rewrite_command(cmd, excluded, transparent_prefixes).is_some()
}

/// Whether an already-`rtk`-prefixed command counts as coverage. `rtk proxy <cmd>`
/// deliberately runs the raw command unfiltered, so it must not count — that would
/// let the audit flatter itself via its own escape hatch. This is ground truth read
/// directly from the transcript (the model really did invoke `rtk`), not a guess.
fn is_already_rtk(cmd: &str) -> bool {
    let trimmed = cmd.trim();
    trimmed.starts_with("rtk ") && !trimmed.starts_with("rtk proxy")
}

/// Aggregation bucket for supported commands.
struct SupportedBucket {
    rtk_equivalent: &'static str,
    category: &'static str,
    count: usize,
    /// Total estimated tokens *saved* (post-filter). Used for the "Est. Savings" column.
    total_output_tokens: usize,
    /// Total estimated tokens *before* filtering (raw output). Accumulated alongside
    /// `total_output_tokens` so the bucket's effective savings rate can be derived as
    /// `total_output_tokens / total_raw_output_tokens` — a weighted average across
    /// all sub-commands, regardless of which sub-command was seen first.
    total_raw_output_tokens: usize,
    // For display: the most common raw command
    command_counts: HashMap<String, usize>,
}

/// Aggregation bucket for unsupported commands.
struct UnsupportedBucket {
    count: usize,
    example: String,
}

pub fn run(
    project: Option<&str>,
    all: bool,
    since_days: u64,
    limit: usize,
    format: &str,
    verbose: u8,
) -> Result<()> {
    let provider = ClaudeProvider;

    // Determine project filter
    let project_filter = if all {
        None
    } else if let Some(p) = project {
        Some(p.to_string())
    } else {
        // Default: current working directory
        let cwd = std::env::current_dir()?;
        let cwd_str = cwd.to_string_lossy().to_string();
        let encoded = ClaudeProvider::encode_project_path(&cwd_str);
        Some(encoded)
    };

    let sessions = provider.discover_sessions(project_filter.as_deref(), Some(since_days))?;

    if verbose > 0 {
        eprintln!("Scanning {} session files...", sessions.len());
        for s in &sessions {
            eprintln!("  {}", s.display());
        }
    }

    // Transcripts record commands as the model emitted them, before the PreToolUse
    // hook rewrites them. Prefer ground truth from the `hook_decisions` log (keyed by
    // `tool_use_id`, populated at the moment the hook actually ran) over guessing from
    // today's hook-install state; fall back to a heuristic only for history that
    // predates logging (or isn't Claude Code).
    let hook_installed = hook_status() != HookStatus::Missing;
    let (excluded, transparent_prefixes) = crate::core::config::Config::load()
        .map(|c| (c.hooks.exclude_commands, c.hooks.transparent_prefixes))
        .unwrap_or_default();

    let cutoff = Utc::now() - chrono::Duration::days(since_days as i64);
    let (hook_log, measured_since): (HashMap<String, HookDecisionRecord>, Option<DateTime<Utc>>) =
        Tracker::new()
            .map(|t| {
                let log = t.hook_decisions_since(cutoff).unwrap_or_default();
                let measured_since = t.earliest_hook_decision_timestamp().ok().flatten();
                (log, measured_since)
            })
            .unwrap_or_default();

    let mut total_commands: usize = 0;
    let mut already_rtk: usize = 0;
    let mut already_rtk_estimated: usize = 0;
    let mut parse_errors: usize = 0;
    let mut rtk_disabled_count: usize = 0;
    let mut rtk_disabled_estimated: usize = 0;
    let mut rtk_disabled_cmds: HashMap<String, usize> = HashMap::new();
    let mut supported_map: HashMap<&'static str, SupportedBucket> = HashMap::new();
    let mut unsupported_map: HashMap<String, UnsupportedBucket> = HashMap::new();

    for session_path in &sessions {
        let extracted = match provider.extract_commands(session_path) {
            Ok(cmds) => cmds,
            Err(e) => {
                if verbose > 0 {
                    eprintln!("Warning: skipping {}: {}", session_path.display(), e);
                }
                parse_errors += 1;
                continue;
            }
        };

        for ext_cmd in &extracted {
            let parts = split_command_chain(&ext_cmd.command);
            for part in parts {
                total_commands += 1;

                // Detect RTK_DISABLED= bypass before classification
                let (env_prefix, actual_cmd) = strip_disabled_prefix(part);
                if prefix_contains_rtk_disabled(env_prefix) {
                    // Only count if the underlying command is one RTK supports, AND the
                    // hook would actually have covered it — otherwise RTK_DISABLED=
                    // bypassed nothing (hook wasn't installed / excluded / would defer
                    // / would deny), and flagging it as a "bypass" would be false advice.
                    if let Classification::Supported { .. } = classify_command(actual_cmd) {
                        let coverage = hook_coverage(
                            actual_cmd,
                            &ext_cmd.tool_use_id,
                            &hook_log,
                            hook_installed,
                            &excluded,
                            &transparent_prefixes,
                        );
                        if coverage.is_covered() {
                            rtk_disabled_count += 1;
                            if coverage.is_estimated() {
                                rtk_disabled_estimated += 1;
                            }
                            let display = truncate_command(actual_cmd);
                            *rtk_disabled_cmds.entry(display).or_insert(0) += 1;
                        }
                    }
                    continue;
                }

                match classify_command(part) {
                    Classification::Supported {
                        rtk_equivalent,
                        category,
                        estimated_savings_pct,
                        status,
                    } => {
                        let coverage = hook_coverage(
                            part,
                            &ext_cmd.tool_use_id,
                            &hook_log,
                            hook_installed,
                            &excluded,
                            &transparent_prefixes,
                        );

                        if coverage.is_covered() {
                            // Hook already routed this through RTK at runtime — it's
                            // coverage, not a missed opportunity.
                            already_rtk += 1;
                            if coverage.is_estimated() {
                                already_rtk_estimated += 1;
                            }
                            continue;
                        }

                        let bucket = supported_map.entry(rtk_equivalent).or_insert_with(|| {
                            SupportedBucket {
                                rtk_equivalent,
                                category,
                                count: 0,
                                total_output_tokens: 0,
                                total_raw_output_tokens: 0,
                                command_counts: HashMap::new(),
                            }
                        });

                        bucket.count += 1;

                        // Estimate tokens for this command
                        let output_tokens = if let Some(len) = ext_cmd.output_len {
                            // Real: from tool_result content length
                            len / 4
                        } else {
                            // Fallback: category average
                            let subcmd = extract_subcmd(part);
                            category_avg_tokens(category, subcmd)
                        };

                        let savings =
                            (output_tokens as f64 * estimated_savings_pct / 100.0) as usize;
                        bucket.total_output_tokens += savings;
                        // Accumulate pre-savings tokens so we can compute a weighted effective
                        // savings rate across all sub-commands in this bucket later.
                        bucket.total_raw_output_tokens += output_tokens;

                        // Track the display name with status
                        let display_name = truncate_command(part);
                        let entry = bucket
                            .command_counts
                            .entry(format!("{}:{:?}", display_name, status))
                            .or_insert(0);
                        *entry += 1;
                    }
                    Classification::Unsupported { base_command } => {
                        let bucket = unsupported_map.entry(base_command).or_insert_with(|| {
                            UnsupportedBucket {
                                count: 0,
                                example: part.to_string(),
                            }
                        });
                        bucket.count += 1;
                    }
                    Classification::Ignored => {
                        // Ground truth from the transcript itself — the model really
                        // did invoke `rtk` directly (excluding the deliberate
                        // unfiltered `rtk proxy` escape hatch).
                        if is_already_rtk(part) {
                            already_rtk += 1;
                        }
                        // Otherwise just skip
                    }
                }
            }
        }
    }

    // Build report
    let mut supported: Vec<SupportedEntry> = supported_map
        .into_values()
        .map(|bucket| {
            // Pick the most common command as the display name
            let (command_with_status, status) = bucket
                .command_counts
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(name, _)| {
                    // Extract status from "command:Status" format
                    if let Some(colon_pos) = name.rfind(':') {
                        let cmd = name[..colon_pos].to_string();
                        let status_str = &name[colon_pos + 1..];
                        let status = match status_str {
                            "Passthrough" => report::RtkStatus::Passthrough,
                            "NotSupported" => report::RtkStatus::NotSupported,
                            _ => report::RtkStatus::Existing,
                        };
                        (cmd, status)
                    } else {
                        (name, report::RtkStatus::Existing)
                    }
                })
                .unwrap_or_else(|| (String::new(), report::RtkStatus::Existing));

            // Derive the effective savings rate from accumulated totals rather than
            // using the first-seen sub-command's rate. This gives a weighted average
            // across all sub-commands that fell in this bucket.
            let effective_savings_pct = if bucket.total_raw_output_tokens > 0 {
                bucket.total_output_tokens as f64 * 100.0 / bucket.total_raw_output_tokens as f64
            } else {
                0.0
            };

            SupportedEntry {
                command: command_with_status,
                count: bucket.count,
                rtk_equivalent: bucket.rtk_equivalent,
                category: bucket.category,
                estimated_savings_tokens: bucket.total_output_tokens,
                estimated_savings_pct: effective_savings_pct,
                rtk_status: status,
            }
        })
        .collect();

    // Sort by estimated savings descending
    supported.sort_by_key(|b| std::cmp::Reverse(b.estimated_savings_tokens));

    let mut unsupported: Vec<UnsupportedEntry> = unsupported_map
        .into_iter()
        .map(|(base, bucket)| UnsupportedEntry {
            base_command: base,
            count: bucket.count,
            example: bucket.example,
        })
        .collect();

    // Sort by count descending
    unsupported.sort_by_key(|b| std::cmp::Reverse(b.count));

    // Build RTK_DISABLED examples sorted by frequency (top 5)
    let rtk_disabled_examples: Vec<String> = {
        let mut sorted: Vec<_> = rtk_disabled_cmds.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted
            .into_iter()
            .take(5)
            .map(|(cmd, count)| format!("{} ({}x)", cmd, count))
            .collect()
    };

    let report = DiscoverReport {
        sessions_scanned: sessions.len(),
        total_commands,
        already_rtk,
        already_rtk_estimated,
        measured_since: measured_since.map(|ts| ts.format("%Y-%m-%d").to_string()),
        since_days,
        supported,
        unsupported,
        parse_errors,
        rtk_disabled_count,
        rtk_disabled_estimated,
        rtk_disabled_examples,
        agent_status: report::AgentIntegrationStatus::detect(),
    };

    match format {
        "json" => println!("{}", report::format_json(&report)),
        _ => print!("{}", report::format_text(&report, limit, verbose > 0)),
    }

    Ok(())
}

/// Extract the subcommand from a command string (second word).
fn extract_subcmd(cmd: &str) -> &str {
    let parts: Vec<&str> = cmd.trim().splitn(3, char::is_whitespace).collect();
    if parts.len() >= 2 {
        parts[1]
    } else {
        ""
    }
}

/// Truncate a command for display (keep first meaningful portion).
fn truncate_command(cmd: &str) -> String {
    let trimmed = cmd.trim();
    // Keep first two words for display
    let parts: Vec<&str> = trimmed.splitn(3, char::is_whitespace).collect();
    match parts.len() {
        0 => String::new(),
        1 => parts[0].to_string(),
        _ => format!("{} {}", parts[0], parts[1]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tracking::HookOutcome;

    // rtk-ai/rtk#3148: hook-rewritten commands were counted as missed savings
    // because transcripts record the pre-rewrite (raw) form of the command.
    // Real hook decisions are logged (`hook_decisions`, keyed by `tool_use_id`)
    // now so coverage is measured, not guessed; `estimate_hook_coverage`/
    // `Coverage::Estimated` — partially modeled on #3164's heuristic, with its
    // permission-deny gap fixed — remain only as the fallback for history that
    // predates that log.

    fn record(decision: HookOutcome) -> HookDecisionRecord {
        HookDecisionRecord { decision }
    }

    #[test]
    fn test_estimate_hook_coverage_false_when_hook_not_installed() {
        // No hook installed → the raw command genuinely ran unfiltered; still a miss.
        assert!(!estimate_hook_coverage_with_verdict(
            "grep -n foo bar.py",
            PermissionVerdict::Default,
            false,
            &[],
            &[]
        ));
    }

    #[test]
    fn test_estimate_hook_coverage_true_for_rewritable_command() {
        assert!(estimate_hook_coverage_with_verdict(
            "grep -n foo bar.py",
            PermissionVerdict::Default,
            true,
            &[],
            &[]
        ));
        assert!(estimate_hook_coverage_with_verdict(
            "ls -la",
            PermissionVerdict::Allow,
            true,
            &[],
            &[]
        ));
    }

    #[test]
    fn test_estimate_hook_coverage_false_for_unattestable_construct() {
        // The hook itself defers on unattestable constructs (substitutions, etc.),
        // so a command containing one was never actually rewritten — genuine miss.
        assert!(!estimate_hook_coverage_with_verdict(
            "git status $(rm -rf /tmp/x)",
            PermissionVerdict::Default,
            true,
            &[],
            &[]
        ));
    }

    #[test]
    fn test_estimate_hook_coverage_false_when_excluded_by_config() {
        // Even with the hook installed, a command the user excluded via config
        // was never rewritten — the hook stepped aside, so it's a genuine miss.
        let excluded = vec!["grep".to_string()];
        assert!(!estimate_hook_coverage_with_verdict(
            "grep -n foo bar.py",
            PermissionVerdict::Default,
            true,
            &excluded,
            &[]
        ));
    }

    #[test]
    fn test_estimate_hook_coverage_false_when_denied() {
        // A command matching a permissions.deny rule is never auto-rewritten by the
        // real hook (it defers to Claude Code's native deny handling) — counting it
        // as covered would over-count savings for commands the hook actually denied.
        assert!(!estimate_hook_coverage_with_verdict(
            "rm -rf /",
            PermissionVerdict::Deny,
            true,
            &[],
            &[]
        ));
    }

    #[test]
    fn test_is_already_rtk_plain_rewrite() {
        assert!(is_already_rtk("rtk grep -n foo bar.py"));
    }

    #[test]
    fn test_is_already_rtk_excludes_proxy_escape_hatch() {
        // rtk#3148 secondary finding: `rtk proxy` deliberately bypasses filtering,
        // so it must not count as coverage.
        assert!(!is_already_rtk("rtk proxy grep -n foo bar.py"));
    }

    #[test]
    fn test_is_already_rtk_false_for_unrelated_command() {
        assert!(!is_already_rtk("grep -n foo bar.py"));
    }

    #[test]
    fn test_hook_coverage_prefers_measured_log_over_estimate() {
        // Even if the current hook state would estimate "not covered" (hook
        // uninstalled today), a real log row is ground truth and wins.
        let mut log = HashMap::new();
        log.insert("toolu_1".to_string(), record(HookOutcome::Allow));

        let coverage = hook_coverage("git status", "toolu_1", &log, false, &[], &[]);
        assert!(matches!(coverage, Coverage::Measured(true)));
        assert!(coverage.is_covered());
        assert!(!coverage.is_estimated());
    }

    #[test]
    fn test_hook_coverage_measured_deny_is_a_genuine_miss() {
        let mut log = HashMap::new();
        log.insert("toolu_1".to_string(), record(HookOutcome::Deny));

        let coverage = hook_coverage("rm -rf /", "toolu_1", &log, true, &[], &[]);
        assert!(matches!(coverage, Coverage::Measured(false)));
        assert!(!coverage.is_covered());
    }

    #[test]
    fn test_hook_coverage_falls_back_to_estimate_when_no_log_row() {
        // No log entry for this tool_use_id — predates logging (or non-Claude) —
        // falls back to the current-state heuristic, flagged as estimated.
        let log = HashMap::new();

        let coverage = hook_coverage("ls -la", "toolu_missing", &log, true, &[], &[]);
        assert!(coverage.is_estimated());
    }
}
