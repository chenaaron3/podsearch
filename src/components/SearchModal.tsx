import { Search as SearchIcon } from 'lucide-react';
import { useState } from 'react';
import YouTube from 'react-youtube';
import { Badge } from '~/components/ui/badge';
import { Button } from '~/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '~/components/ui/card';
import { Textarea } from '~/components/ui/textarea';

interface SearchModalProps {
    title: string;
    description: string;
    placeholder: string;
    onSearch: (query: string) => void;
    isSearching: boolean;
    loadingMessage: string;
    buttonText: string;
    youtubeId?: string;
    disabled?: boolean;
}

export function SearchModal({
    title,
    description,
    placeholder,
    onSearch,
    isSearching,
    loadingMessage,
    buttonText,
    youtubeId,
    disabled = false,
}: SearchModalProps) {
    const [searchQuery, setSearchQuery] = useState('');

    const handleSearch = () => {
        if (!searchQuery.trim()) return;
        onSearch(searchQuery);
    };

    return (
        <div className="max-w-3xl mx-auto px-2 sm:px-0">
            <Card className="shadow-xl border-0 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
                <CardHeader className="text-center pb-4 sm:pb-6">
                    <CardTitle className="text-xl sm:text-2xl">{title}</CardTitle>
                    <CardDescription className="text-sm sm:text-base">{description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4 sm:space-y-6">
                    <div className="space-y-4">
                        {youtubeId && (
                            <div className="flex justify-center">
                                <div className="aspect-video rounded-lg overflow-hidden bg-black relative shadow-lg w-full max-w-2xl">
                                    <YouTube
                                        videoId={youtubeId}
                                        className="w-full h-full"
                                        key={`${youtubeId}`}
                                    />
                                </div>
                            </div>
                        )}

                        <Textarea
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder={placeholder}
                            className="min-h-[100px] sm:min-h-[120px] resize-none text-sm sm:text-base"
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    void handleSearch();
                                }
                            }}
                        />

                        <Button
                            onClick={() => void handleSearch()}
                            disabled={!searchQuery.trim() || isSearching || disabled}
                            size="lg"
                            className="w-full h-10 sm:h-12 text-base sm:text-lg font-semibold"
                        >
                            {isSearching ? (
                                <>
                                    <div className="animate-spin rounded-full h-4 w-4 sm:h-5 sm:w-5 border-b-2 border-white mr-2" />
                                    <span className="animate-pulse text-sm sm:text-base">{loadingMessage}</span>
                                </>
                            ) : (
                                <>
                                    <SearchIcon className="mr-2 h-4 w-4 sm:h-5 sm:w-5" />
                                    {buttonText}
                                </>
                            )}
                        </Button>
                    </div>

                    <div className="text-center text-xs sm:text-sm text-muted-foreground">
                        <p>💡 Try searching for topics like:</p>
                        <div className="flex flex-wrap justify-center gap-1.5 sm:gap-2 mt-2">
                            <Badge variant="secondary" className="text-xs">startup advice</Badge>
                            <Badge variant="secondary" className="text-xs">leadership</Badge>
                            <Badge variant="secondary" className="text-xs">personal growth</Badge>
                            <Badge variant="secondary" className="text-xs">business strategy</Badge>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
} 