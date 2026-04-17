export const REPO_URL = 'https://github.com/Charbel199/gpu-anatomy';
export const REPO_BRANCH = 'main';

export function githubBlobUrl(path: string): string {
  return `${REPO_URL}/blob/${REPO_BRANCH}/${path}`;
}
