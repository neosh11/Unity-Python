using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using NetMQ;
using NetMQ.Sockets;

using UnityEngine;

namespace Freelance.ModelOne.Client
{
    internal class FreelanceProgram
    {

        private const int RequestTimeout = 1000; // ms

        RequestSocket client;

        // For testing purposes, we will keep this number low
        // - need to figure outa way to run this asynchronously rather than in sync
        // = As it will obviosly cause freexing problems

        private const int MaxRetries = 1; // Before we abandon

        private static List<string> endpoints = new List<string>
            {
                "tcp://localhost:5555"
            };

        public string response;
        public async Task<byte[]> Request(byte[] data)
        {
            try
            {
                if (endpoints.Count == 1)
                {
                    for (int i = 0; i < MaxRetries; i++)
                    {

                        Debug.Log(i);
                        if (await TryRequest(endpoints[0], data))
                            break;
                    }
                }
                else if (endpoints.Count > 1)
                {
                    foreach (string endpoint in endpoints)
                    {
                        if (await TryRequest(endpoint, data))
                            break;
                    }
                }
                else
                {
                    Debug.Log("No endpoints");
                }
            }
            catch (Exception e)
            {
                Debug.Log(e);
            }

            return null;
        }

        private async Task<bool> TryRequest(string endpoint, byte[] requestString)
        {
            AsyncIO.ForceDotNet.Force();

            bool pollResult = false;
            try
            {
                Debug.Log($"Trying echo service at {endpoint}");

                using (client = new RequestSocket())
                {
                    client.Options.Linger = TimeSpan.Zero;

                    client.Connect(endpoint);

                    client.TrySendFrame(TimeSpan.FromSeconds(5), requestString);

                    client.ReceiveReady += ClientOnReceiveReady;

                    pollResult = await TryReceive(client);
                    Debug.Log("HELLO??");

                    client.ReceiveReady -= ClientOnReceiveReady;

                    client.Disconnect(endpoint);
                }
            }

            catch (Exception e)
            {
                Debug.Log(e);
            }

            return pollResult;
        }

        // Prevent blocking

        private async Task<bool> TryReceive(RequestSocket client)
        {

            var count = 10;
            var pollResult = false;

            for (int i = 0; i < count; i++)
            {
                pollResult = client.Poll(TimeSpan.FromMilliseconds(100));
                if (pollResult)
                {
                    break;
                }
                // wait for a bit
                await Task.Delay(5000);
            }
            return pollResult;
        }

        private void ClientOnReceiveReady(object sender, NetMQSocketEventArgs args)
        {
            var res = args.Socket.ReceiveFrameString();
            Debug.Log($"Server replied ({res})");

            response = res;
        }
    }
}