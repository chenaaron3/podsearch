import Head from 'next/head';
import Link from 'next/link';
import { Badge } from '~/components/ui/badge';
import { Button } from '~/components/ui/button';
import { Card, CardDescription, CardHeader, CardTitle } from '~/components/ui/card';

export default function Home() {
  return (
    <>
      <Head>
        <title>Diary of a CEO Search</title>
        <meta name="description" content="AI-powered search through Diary of a CEO podcast episodes" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="min-h-screen bg-background flex items-center justify-center">
        <div className="container mx-auto px-4 py-8">
          {/* Hero Section */}
          <div className="flex flex-col items-center justify-center text-center space-y-6 mb-12">
            <Badge variant="secondary" className="mb-2">
              AI-Powered Podcast Search
            </Badge>

            <h1 className="text-3xl font-bold tracking-tight text-foreground sm:text-5xl lg:text-6xl">
              <span className="bg-gradient-to-r from-primary via-purple-500 to-blue-500 bg-[length:200%_100%] bg-clip-text text-transparent animate-generating">
                Diary of a CEO
              </span>{' '}
              Search
            </h1>

            <p className="max-w-2xl text-base text-muted-foreground leading-relaxed">
              Find the perfect clips from Steven Bartlett&apos;s podcast using AI-powered search.
              Tell us what you&apos;re interested in, and we&apos;ll find exactly what you&apos;re looking for.
            </p>

            <div className="flex flex-col sm:flex-row gap-4 pt-2">
              <Button asChild size="lg" className="text-base px-8 py-6">
                <Link href="/search">
                  Start Searching
                  <svg className="w-5 h-5 ml-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
                  </svg>
                </Link>
              </Button>
            </div>
          </div>

          {/* Features Section */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto mb-8">
            <Card className="border-border/50">
              <CardHeader className="pb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                  <svg className="w-5 h-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <CardTitle className="text-lg">Smart Search</CardTitle>
                <CardDescription>
                  Describe what you&apos;re looking for in natural language and our AI will understand your intent.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50">
              <CardHeader className="pb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                  <svg className="w-5 h-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
                  </svg>
                </div>
                <CardTitle className="text-lg">Perfect Clips</CardTitle>
                <CardDescription>
                  Get personalized video clips that jump straight to the most relevant moments from thousands of hours of content.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-border/50">
              <CardHeader className="pb-4">
                <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center mb-3">
                  <svg className="w-5 h-5 text-primary" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
                  </svg>
                </div>
                <CardTitle className="text-lg">All Episodes</CardTitle>
                <CardDescription>
                  Search through every episode of Diary of a CEO to find exactly what you&apos;re looking for, no matter when it was released.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>

          {/* Footer */}
          <div className="text-center">
            <p className="text-sm text-muted-foreground">
              Powered by AI semantic search through thousands of hours of podcast content
            </p>
          </div>
        </div>
      </main>
      <style jsx>{`
        @keyframes generating {
          0%, 100% {
            background-position: 0% 50%;
          }
          25% {
            background-position: 100% 50%;
          }
          50% {
            background-position: 100% 50%;
          }
          75% {
            background-position: 0% 50%;
          }
        }
        
        .animate-generating {
          animation: generating 2s ease-in-out infinite;
        }
      `}</style>
    </>
  );
}
