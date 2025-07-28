import { create } from 'zustand';
import { persist } from 'zustand/middleware';

import type { SearchSegment } from "~/types";

interface SearchState {
  // Global search state
  lastGlobalQuery: string;
  lastGlobalResults: SearchSegment[];

  // Actions
  setLastGlobalSearch: (query: string, results: SearchSegment[]) => void;
  clearGlobalSearch: () => void;
}

export const useSearchStore = create<SearchState>()(
  persist(
    (set) => ({
      // Initial state
      lastGlobalQuery: "",
      lastGlobalResults: [],

      // Actions
      setLastGlobalSearch: (query: string, results: SearchSegment[]) =>
        set({
          lastGlobalQuery: query,
          lastGlobalResults: results,
        }),

      clearGlobalSearch: () =>
        set({
          lastGlobalQuery: "",
          lastGlobalResults: [],
        }),
    }),
    {
      name: "search-storage",
      partialize: (state) => ({
        lastGlobalQuery: state.lastGlobalQuery,
        lastGlobalResults: state.lastGlobalResults,
      }),
    },
  ),
);
