"use client";

interface Props {
  query: string;
  onReset: () => void;
  className: string;
  buttonClassName: string;
}

export function QueueEmpty({ query, onReset, className, buttonClassName }: Props) {
  const trimmed = query.trim();
  return (
    <div className={className}>
      {trimmed ? (
        <>No leads match &ldquo;{trimmed}&rdquo;. </>
      ) : (
        <>No leads match these filters. </>
      )}
      <button type="button" className={buttonClassName} onClick={onReset}>
        Show all
      </button>
    </div>
  );
}
