/**
 * Helper for proxying browser requests to the Modal-hosted FastAPI app.
 * MODAL_BASE_URL is the URL Modal prints after `modal deploy`.
 */
const baseUrl = (): string => {
  const url = process.env.MODAL_BASE_URL;
  if (!url) {
    throw new Error("MODAL_BASE_URL is not set");
  }
  return url.replace(/\/$/, "");
};

export const modalUrl = (path: string): string => `${baseUrl()}${path}`;
