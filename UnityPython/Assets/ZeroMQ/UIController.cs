using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Net.Http.Headers;

using TMPro;

using UnityEngine;
using UnityEngine.UI;
using UnityEditor;

using Transfer;

using NetMQ;
using NetMQ.Sockets;

using Google.Protobuf;


public class UIController : MonoBehaviour
{
    private Transfer.Transfer filesProto;
    private RequestSocket socket;
    private bool running;

    void Start()
    {
        running = true;
    }
    public void OnFileSelect()
    {
        // string path = EditorUtility.OpenFilePanel("Select a folder to upload", "", "*");
        string path = EditorUtility.OpenFolderPanel("Select folder", "", "");
        if (path.Length != 0)
        {
            // loop through directory
            var fileStrings = Directory.GetFiles(path, "*.dcm");
            var length = fileStrings.Length;
            var filesByteData = new List<Google.Protobuf.ByteString>();

            Debug.Log(path);

            for (int i = 0; i < length; i++)
            {
                var d = File.OpenRead(fileStrings[i]);
                filesByteData.Add(Google.Protobuf.ByteString.FromStream(d));
            }

            // To Proto
            filesProto = ConvertFilesByteToProto(filesByteData, Transfer.Transfer.Types.TypeFunction.BedRemoval);
            // Make request?

            Request();

            Debug.Log("requested");
        }
        else
        {
            filesProto = null;
        }
    }


    Transfer.Transfer ConvertFilesByteToProto(List<Google.Protobuf.ByteString> filesByteData, Transfer.Transfer.Types.TypeFunction type)
    {
        // To protocol buffer

        if (filesByteData == null)
        {
            Debug.Log("Data could not be loaded");
            return null;
        }

        var data = new Transfer.Transfer
        {
            TypeOfFunction = type
        };
        // convert all byte data to bytestrings

        data.DcmFiles.Add(filesByteData);

        // Transfer protobuf to python.
        return data;
    }


    void Request()
    {
        AsyncIO.ForceDotNet.Force();
        NetMQConfig.Cleanup();

        if (filesProto == null) return;

        // proto to binary
        byte[] outBytes = filesProto.ToByteArray();
        File.WriteAllBytes("C:\\TEST\\haha", outBytes);

        Transfer.Transfer inMsg;
        socket = new RequestSocket("tcp://localhost:5559");
        socket.SendFrame(outBytes);
        while (running)
        {
            byte[] msg;
            var received = socket.TryReceiveFrameBytes(TimeSpan.FromSeconds(1), out msg);
            if (received)
            {
                inMsg = Transfer.Transfer.Parser.ParseFrom(msg);
                break;
            }
            if (!received && !running) return;
        }

        socket.Close();
        NetMQConfig.Cleanup();

        socket = null;
        filesProto = null;
    }

    void OnDestroy()
    {
        Clean();
    }

    void OnScriptsReloaded()
    {
        Clean();
    }

    void Clean()
    {
        running = false;
        if (socket != null)
        {
            socket.Close();
            socket.Dispose();
        }
        NetMQConfig.Cleanup();
    }

}
