"use client";

import type { LeadRow } from "@/lib/leads";
import styles from "./chat.module.css";

interface Props {
  lead: LeadRow | null;
  expanded: boolean;
  onToggle: () => void;
}

function summaryLine(lead: LeadRow): string {
  const parts = [lead.leadName];
  if (lead.title) parts.push(lead.title);
  if (lead.companyName) parts.push(lead.companyName);
  return parts.filter(Boolean).join(" · ");
}

export function LeadRecordPanel({ lead, expanded, onToggle }: Props) {
  if (!lead) {
    return (
      <div className={styles.panelContext}>
        <span className={styles.panelContextEmpty}>
          Click a lead from the queue to focus the assistant.
        </span>
      </div>
    );
  }

  return (
    <div className={styles.leadRecord}>
      <button
        type="button"
        className={styles.leadRecordHeader}
        onClick={onToggle}
        aria-expanded={expanded}
      >
        <span className={styles.leadRecordCaret} aria-hidden="true">
          {expanded ? "▼" : "▸"}
        </span>
        <span className={styles.leadRecordTitle}>{summaryLine(lead)}</span>
      </button>
      {expanded && (
        <div className={styles.leadRecordBody}>
          <div>
            <div className={styles.leadRecordLabel}>Profile</div>
            <pre className={styles.recordText}>{lead.profile}</pre>
          </div>
          {lead.company && (
            <div>
              <div className={styles.leadRecordLabel}>Company</div>
              <pre className={styles.recordText}>{lead.company}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
