import type { NextPage } from 'next';
import Head from 'next/head';
import NetworkGraph from '~/components/NetworkGraph';

const NetworkPage: NextPage = () => {
    return (
        <>
            <Head>
                <title>Chapter Network - PodSearch</title>
                <meta name="description" content="Interactive network visualization with hierarchical clustering" />
            </Head>

            <div className="w-screen h-screen">
                <NetworkGraph />
            </div>
        </>
    );
};

export default NetworkPage; 