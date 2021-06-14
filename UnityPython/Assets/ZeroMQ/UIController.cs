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

using MDPClientExample;
using Freelance.ModelOne.Client;


public class UIController : MonoBehaviour
{
    private Transfer.Transfer filesProto;
    private RequestSocket socket;
    private bool running;

    private FreelanceProgram freelanceProgram;

    void Start()
    {
        running = true;
    }

    void Update()
    {
        if (freelanceProgram != null)
        {
            if (freelanceProgram.response != null)
            {
                Debug.Log($"Unity had the following responsne :- {freelanceProgram.response}");
                freelanceProgram = null;

                // Display this Dicom :) freelanceProgram.response
            }
        }
    }
    public async void OnFileSelect()
    {
        // string path = EditorUtility.OpenFilePanel("Select a folder to upload", "", "*");
        string path = EditorUtility.OpenFolderPanel("Select folder", "Desktop", "");
        if (path.Length != 0)
        {
            Debug.Log(path);

            // Send location to python.

            Byte[] b = Encoding.ASCII.GetBytes(path);
            Debug.Log("Requesting");

            freelanceProgram = new FreelanceProgram();
            await freelanceProgram.Request(b);

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




    void OnDestroy()
    {
        Clean();
    }

    void OnScriptsReloaded()
    {
        Clean();
    }

    void OnApplicationQuit()
    {
        Clean();
    }
    void Clean()
    {
        running = false;
        NetMQConfig.Cleanup(false);
    }

}
