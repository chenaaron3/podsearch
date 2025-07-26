import Head from 'next/head';
import Link from 'next/link';

export default function Home() {
  return (
    <>
      <Head>
        <title>Diary of a CEO Search</title>
        <meta name="description" content="AI-powered search through Diary of a CEO podcast episodes" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        <div className="container flex flex-col items-center justify-center gap-12 px-4 py-16 text-center">
          <h1 className="text-6xl font-extrabold tracking-tight text-white sm:text-[5rem]">
            Diary of a <span className="text-[hsl(280,100%,70%)]">CEO</span> Search
          </h1>

          <p className="text-xl text-slate-300 max-w-2xl leading-relaxed">
            Find the perfect clips from Steven Bartlett&apos;s podcast using AI-powered search.
            Tell us what you&apos;re interested in, and we&apos;ll ask clarifying questions to find
            exactly what you&apos;re looking for.
          </p>

          <div className="flex flex-col sm:flex-row gap-4">
            <Link
              href="/search"
              className="flex items-center justify-center px-8 py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold rounded-xl hover:from-purple-700 hover:to-blue-700 transition-all text-lg"
            >
              Start Searching
              <svg className="w-5 h-5 ml-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
              </svg>
            </Link>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8 max-w-4xl">
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <div className="text-purple-400 mb-3">
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Smart Search</h3>
              <p className="text-slate-300 text-sm">
                Describe what you&apos;re looking for in natural language and our AI will understand your intent.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <div className="text-blue-400 mb-3">
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clipRule="evenodd" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Clarifying Questions</h3>
              <p className="text-slate-300 text-sm">
                We&apos;ll ask follow-up questions to understand exactly what type of content you want.
              </p>
            </div>

            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6 border border-white/20">
              <div className="text-green-400 mb-3">
                <svg className="w-8 h-8" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Perfect Clips</h3>
              <p className="text-slate-300 text-sm">
                Get 5 personalized video clips that jump straight to the most relevant moments.
              </p>
            </div>
          </div>

          <div className="text-slate-400 text-sm mt-8">
            Powered by AI semantic search through thousands of hours of podcast content
          </div>
        </div>
      </main>
    </>
  );
}
