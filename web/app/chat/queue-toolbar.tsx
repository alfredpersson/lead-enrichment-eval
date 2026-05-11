"use client";

import { STATUS_LABELS, STATUS_ORDER, type LeadStatus } from "@/lib/leads";
import styles from "./chat.module.css";

export type SortBy = "recent" | "name" | "status";

const SORT_LABELS: Record<SortBy, string> = {
  recent: "Most recent",
  name: "Name",
  status: "Status",
};

interface Props {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  statusFilters: Set<LeadStatus>;
  onToggleStatus: (s: LeadStatus) => void;
  sortBy: SortBy;
  onSortChange: (s: SortBy) => void;
  selectedCount: number;
  allVisibleSelected: boolean;
  onSelectAllVisible: () => void;
  onClearSelection: () => void;
  onSetStatusForSelected: (s: LeadStatus) => void;
  onDeleteSelected: () => void;
}

export function QueueToolbar({
  searchQuery,
  onSearchChange,
  statusFilters,
  onToggleStatus,
  sortBy,
  onSortChange,
  selectedCount,
  allVisibleSelected,
  onSelectAllVisible,
  onClearSelection,
  onSetStatusForSelected,
  onDeleteSelected,
}: Props) {
  return (
    <div className={styles.queueControls}>
      <div className={styles.queueControlsRow}>
        <input
          type="search"
          className={styles.searchInput}
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Search by name, title, company…"
          aria-label="Search leads"
        />
        <select
          className={styles.sortSelect}
          value={sortBy}
          onChange={(e) => onSortChange(e.target.value as SortBy)}
          aria-label="Sort leads"
        >
          {(Object.keys(SORT_LABELS) as SortBy[]).map((s) => (
            <option key={s} value={s}>
              Sort: {SORT_LABELS[s]}
            </option>
          ))}
        </select>
      </div>

      <div className={styles.statusChips} role="group" aria-label="Filter by status">
        {STATUS_ORDER.map((s) => {
          const active = statusFilters.has(s);
          return (
            <button
              key={s}
              type="button"
              className={`${styles.statusChip} ${
                active ? styles.statusChipActive : ""
              }`}
              onClick={() => onToggleStatus(s)}
              aria-pressed={active}
            >
              {STATUS_LABELS[s]}
            </button>
          );
        })}
      </div>

      {selectedCount > 0 && (
        <div className={styles.bulkBar} role="region" aria-label="Bulk actions">
          <div className={styles.bulkBarLeft}>
            <span>
              {selectedCount} selected
            </span>
            <button
              type="button"
              className={styles.bulkLink}
              onClick={
                allVisibleSelected ? onClearSelection : onSelectAllVisible
              }
            >
              {allVisibleSelected ? "Clear selection" : "Select all visible"}
            </button>
          </div>
          <div className={styles.bulkBarRight}>
            <select
              className={styles.bulkSelect}
              value=""
              onChange={(e) => {
                const v = e.target.value as LeadStatus | "";
                if (v) onSetStatusForSelected(v);
                e.target.value = "";
              }}
              aria-label="Set status for selected"
            >
              <option value="" disabled>
                Set status…
              </option>
              {STATUS_ORDER.map((s) => (
                <option key={s} value={s}>
                  {STATUS_LABELS[s]}
                </option>
              ))}
            </select>
            <button
              type="button"
              className={styles.bulkDelete}
              onClick={onDeleteSelected}
            >
              Delete
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
