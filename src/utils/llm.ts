import { readFileSync } from 'fs';
import { join } from 'path';

// Load prompts at build time
export const RANK_PROMPT = readFileSync(
  join(process.cwd(), "src", "prompts", "rank_relevance.txt"),
  "utf-8",
);
export const SEEK_PROMPT = readFileSync(
  join(process.cwd(), "src", "prompts", "seek_clip.txt"),
  "utf-8",
);
export const GRAMMER_PROMPT = readFileSync(
  join(process.cwd(), "src", "prompts", "fix_grammer.txt"),
  "utf-8",
);

// Helper function to replace placeholders in prompts
export function replacePromptPlaceholders(
  prompt: string,
  replacements: Record<string, string>,
): string {
  let result = prompt;
  for (const [placeholder, value] of Object.entries(replacements)) {
    result = result.replace(
      new RegExp(`\\{\\{${placeholder}\\}\\}`, "g"),
      value,
    );
  }
  return result;
}
