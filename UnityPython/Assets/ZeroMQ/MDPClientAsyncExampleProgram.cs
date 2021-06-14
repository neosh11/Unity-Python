using System;
using System.Diagnostics;
using System.Threading;
using MajordomoProtocol;
using NetMQ;

namespace MDPClientExample
{
    internal static class MDPClientAsyncExampleProgram
    {
        /// <summary>
        ///     usage:  MDPClientAsyncExample [-v] [-rn] (1 ;lt n ;lt 100000 / Default == 10)
        /// 
        ///     implements a MDPClientAsync API usage
        /// </summary>
        public static void Main(string service)
        {

            AsyncIO.ForceDotNet.Force();


            const string service_name = "echo";
            const int max_runs = 100000;

            bool verbose = false;
            int runs = 10;


            runs = runs == -1 ? 10 : runs > max_runs ? max_runs : runs;

            var id = new[] { (byte)'C', (byte)'1' };

            var watch = new Stopwatch();

            Console.WriteLine("Starting MDPClient and will send {0} requests to service <{1}>.", runs, service_name);
            Console.WriteLine("(writes '.' for every 100 and '|' for every 1000 requests)\n");

            try
            {
                // create MDP client and set verboseness && use automatic disposal
                using (var session = new MDPClientAsync("tcp://localhost:5555", id))
                {

                    if (verbose)
                        session.LogInfoReady += (s, e) => Console.WriteLine("{0}", e.Info);

                    session.ReplyReady += (s, e) => Console.WriteLine("{0}", e.Reply);

                    // just give everything time to settle in
                    Thread.Sleep(500);

                    watch.Start();

                    for (int count = 0; count < runs; count++)
                    {
                        var request = new NetMQMessage();
                        // set the request data
                        request.Push("Hello World!");
                        // send the request to the service
                        session.Send(service_name, request);

                        if (count % 1000 == 0)
                            Console.Write("|");
                        else
                            if (count % 100 == 0)
                            Console.Write(".");
                    }

                    watch.Stop();
                    Console.Write("\nStop receiving with any key!");
                    Console.ReadKey();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine("ERROR:");
                Console.WriteLine(ex.Message);
                Console.WriteLine(ex.StackTrace);
                NetMQConfig.Cleanup();

                return;
            }

            var time = watch.Elapsed;
            NetMQConfig.Cleanup();


            Console.WriteLine("{0} request/replies in {1} ms processed! Took {2:N3} ms per REQ/REP",
                runs,
                time.TotalMilliseconds,
                time.TotalMilliseconds / runs);

            Console.Write("\nExit with any key!");
            Console.ReadKey();
        }

        private static int GetInt(string s)
        {
            var num = s.Remove(0, 2);

            int runs;
            var success = int.TryParse(num, out runs);

            return success ? runs : -1;
        }
    }
}
