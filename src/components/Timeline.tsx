import type { SearchSegment } from '~/types';

interface TimelineProps {
    duration: number;
    clips: SearchSegment[];
    currentTime: number;
    onClipSelect: (clip: SearchSegment) => void;
    onTimeUpdate?: (time: number) => void;
    className?: string;
}

export function Timeline({
    duration,
    clips,
    currentTime,
    onClipSelect,
    onTimeUpdate,
    className = '',
}: TimelineProps) {


    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    const getPositionFromTime = (time: number) => {
        return (time / duration) * 100;
    };

    const getTimeFromPosition = (position: number) => {
        return (position / 100) * duration;
    };

    const handleTimelineClick = (event: React.MouseEvent<HTMLDivElement>) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const clickX = event.clientX - rect.left;
        const clickPercentage = (clickX / rect.width) * 100;
        const newTime = getTimeFromPosition(clickPercentage);

        if (onTimeUpdate) {
            onTimeUpdate(newTime);
        }
    };

    const handleClipClick = (event: React.MouseEvent, clip: SearchSegment) => {
        event.stopPropagation();
        onClipSelect(clip);
    };



    return (
        <div className={`relative ${className}`}>
            {/* Timeline Container */}
            <div className="relative bg-slate-200 dark:bg-slate-700 rounded-full h-3 cursor-pointer" onClick={handleTimelineClick}>
                {/* Progress Bar */}
                <div
                    className="absolute top-0 left-0 h-full bg-primary rounded-full transition-all duration-300"
                    style={{ width: `${getPositionFromTime(currentTime)}%` }}
                />



                {/* Clip Markers */}
                {clips.map((clip) => (
                    <div
                        key={clip.id}
                        className="absolute top-1/2 transform -translate-y-1/2 w-4 h-4 cursor-pointer group"
                        style={{ left: `${getPositionFromTime(clip.startTime)}%` }}
                        onClick={(e) => handleClipClick(e, clip)}

                    >
                        {/* Marker */}
                        <div className="w-4 h-4 bg-white dark:bg-slate-800 border-2 border-primary rounded-full shadow-md transition-all duration-200 group-hover:scale-125 group-hover:shadow-lg" />

                        {/* Clip Number */}
                        <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium text-primary bg-white dark:bg-slate-800 px-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                            {formatTime(clip.startTime)}
                        </div>
                    </div>
                ))}

                {/* Playhead */}
                <div
                    className="absolute top-0 w-0.5 h-full bg-red-500 shadow-sm"
                    style={{ left: `${getPositionFromTime(currentTime)}%` }}
                />
            </div>


        </div>
    );
} 