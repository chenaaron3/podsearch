import { Search as SearchIcon } from 'lucide-react';
import { Button } from '~/components/ui/button';

interface SearchBarProps {
    value: string;
    onChange: (value: string) => void;
    onSearch: () => void;
    placeholder?: string;
    isSearching?: boolean;
    disabled?: boolean;
    className?: string;
}

export function SearchBar({
    value,
    onChange,
    onSearch,
    placeholder = "Search for topics, advice, insights...",
    isSearching = false,
    disabled = false,
    className = "",
}: SearchBarProps) {
    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && value.trim() && !disabled) {
            onSearch();
        }
    };

    return (
        <div className={`relative ${className}`}>
            <input
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                className="w-full px-3 sm:px-4 py-2.5 sm:py-3 pl-10 sm:pl-12 pr-16 sm:pr-20 text-base sm:text-lg border border-gray-300 rounded-xl focus:ring-2 focus:ring-primary focus:border-transparent bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm"
                onKeyDown={handleKeyDown}
                disabled={disabled}
            />
            <SearchIcon className="absolute left-3 sm:left-4 top-1/2 transform -translate-y-1/2 h-4 w-4 sm:h-5 sm:w-5 text-gray-400" />
            <Button
                onClick={onSearch}
                disabled={!value.trim() || isSearching || disabled}
                className="absolute right-1.5 sm:right-2 top-1/2 transform -translate-y-1/2 h-8 sm:h-9 px-3 sm:px-4 text-sm"
                size="sm"
            >
                {isSearching ? (
                    <div className="animate-spin rounded-full h-3 w-3 sm:h-4 sm:w-4 border-b-2 border-white" />
                ) : (
                    <span className="hidden sm:inline">Search</span>
                )}
            </Button>
        </div>
    );
} 