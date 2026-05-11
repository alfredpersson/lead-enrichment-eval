// Reads a `text/event-stream` response body and invokes `onEvent` for each
// JSON-decoded `data:` frame. Malformed JSON frames are silently dropped.
// Aborts are surfaced to the caller via the fetch's AbortController.

export async function parseSseStream<T>(
  response: Response,
  onEvent: (event: T) => void,
): Promise<void> {
  if (!response.body) return;
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let sep = buffer.indexOf("\n\n");
    while (sep !== -1) {
      const frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      for (const line of frame.split("\n")) {
        if (!line.startsWith("data:")) continue;
        const json = line.slice(5).trim();
        if (!json) continue;
        try {
          onEvent(JSON.parse(json) as T);
        } catch {
          // malformed frame; skip
        }
      }
      sep = buffer.indexOf("\n\n");
    }
  }
}
