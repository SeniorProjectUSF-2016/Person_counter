//----------------------------------------------------------------------------
//  Copyright (C) 2004-2016 by EMGU Corporation. All rights reserved.       
//----------------------------------------------------------------------------

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.UI;
using Emgu.CV.Cvb;
using Emgu.CV.CvEnum;
using Emgu.CV.VideoSurveillance;
using System.Diagnostics;
using Emgu.CV.Util;
using static Emgu.CV.FileNode;
#if !(__IOS__ || NETFX_CORE)
using Emgu.CV.Cuda;
#endif

namespace VideoSurveilance
{
    public partial class VideoSurveilance : Form
    {
        //private static MCvFont _font = new MCvFont(Emgu.CV.CvEnum.FontType.HersheyPlain, 1.0, 1.0);
        private static Capture _cameraCapture;
        private int counter = 0;
        private static BackgroundSubtractor _fgDetector;
        private static Emgu.CV.Cvb.CvBlobDetector _blobDetector;

        public VideoSurveilance()
        {
            InitializeComponent();
            Run();
        }

        void Run()
        {
            try
            {
                // test for test
                _cameraCapture = new Capture(1);
                _cameraCapture.FlipHorizontal = !_cameraCapture.FlipHorizontal;

            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
                return;
            }

            _fgDetector = new Emgu.CV.VideoSurveillance.BackgroundSubtractorMOG2();
            _blobDetector = new CvBlobDetector();
            //_tracker = new BlobTrackerAuto<Bgr>();

            Application.Idle += ProcessFrame;
        }


        void ProcessFrame(object sender, EventArgs e)
        {
            Mat frame = _cameraCapture.QueryFrame();


            Image<Bgr, Byte> ImageFrame = frame.ToImage<Bgr, Byte>();

            //ImageFrame.ROI = Rectangle.Empty;
            Rectangle roi = new Rectangle(213, 0, 213, 480); // set the roi image.ROI = new Rectangle(x, Y, Width, Height);
            //ImageFrame.ROI = roi;

            ImageFrame.Draw(roi, new Bgr(Color.Green), 5);





            if (ImageFrame != null)   // confirm that image is valid
            {
                Image<Gray, byte> grayframe = ImageFrame.Convert<Gray, byte>();


                long processingTime;
                Rectangle[] results = FindPedestrian.Find(frame, true, out processingTime);
                foreach (Rectangle rect in results)
                    if (1 == 1)
                    {
                        ImageFrame.Draw(rect, new Bgr(Color.Red), 5);

                        ++counter;
                        Console.WriteLine(counter);

                    }


            }

            Mat smoothedFrame = new Mat();
            CvInvoke.GaussianBlur(frame, smoothedFrame, new Size(3, 3), 1); //filter out noises
                                                                            //frame._SmoothGaussian(3); 

            #region use the BG/FG detector to find the forground mask
            Mat forgroundMask = new Mat();
            _fgDetector.Apply(smoothedFrame, forgroundMask);

            #endregion
            CvBlobs blobs = new CvBlobs();
            _blobDetector.Detect(forgroundMask.ToImage<Gray, byte>(), blobs);
            blobs.FilterByArea(1000, int.MaxValue);
            //_tracker.Process(smoothedFrame, forgroundMask);

            foreach (var pair in blobs)
            {
                CvBlob b = pair.Value;
                CvInvoke.Rectangle(frame, b.BoundingBox, new MCvScalar(255.0, 255.0, 255.0), 2);
                //CvInvoke.PutText(frame,  blob.ID.ToString(), Point.Round(blob.Center), FontFace.HersheyPlain, 1.0, new MCvScalar(255.0, 255.0, 255.0));


            }

            imageBox1.Image = ImageFrame;
            //Console.WriteLine(ImageFrame.Size);
            //imageBox2.Image = frame;
            //imageBox2.Image = forgroundMask;
        }

        
    }


    public static class FindPedestrian
    {
        /// <summary>
        /// Find the pedestrian in the image
        /// </summary>
        /// <param name="image">The image</param>
        /// <param name="processingTime">The pedestrian detection time in milliseconds</param>
        /// <returns>The region where pedestrians are detected</returns>
        public static Rectangle[] Find(Mat image, bool tryUseCuda, out long processingTime)
        {
            Stopwatch watch;
            Rectangle[] regions;

#if !(__IOS__ || NETFX_CORE)
            //check if there is a compatible Cuda device to run pedestrian detection
            if (tryUseCuda && CudaInvoke.HasCuda)
            {  //this is the Cuda version
                using (CudaHOG des = new CudaHOG(new Size(64, 128), new Size(16, 16), new Size(8, 8), new Size(8, 8)))
                {
                    des.SetSVMDetector(des.GetDefaultPeopleDetector());

                    watch = Stopwatch.StartNew();
                    using (GpuMat cudaBgr = new GpuMat(image))
                    using (GpuMat cudaBgra = new GpuMat())
                    using (VectorOfRect vr = new VectorOfRect())
                    {
                        CudaInvoke.CvtColor(cudaBgr, cudaBgra, ColorConversion.Bgr2Bgra);
                        des.DetectMultiScale(cudaBgra, vr);
                        regions = vr.ToArray();
                    }
                }
            }
            else
#endif
            {
                //this is the CPU/OpenCL version
                using (HOGDescriptor des = new HOGDescriptor())
                {
                    des.SetSVMDetector(HOGDescriptor.GetDefaultPeopleDetector());

                    //load the image to umat so it will automatically use opencl is available
                    UMat umat = image.ToUMat(AccessType.Read);

                    watch = Stopwatch.StartNew();

                    MCvObjectDetection[] results = des.DetectMultiScale(umat);
                    regions = new Rectangle[results.Length];
                    for (int i = 0; i < results.Length; i++)
                        regions[i] = results[i].Rect;
                    watch.Stop();
                }
            }

            processingTime = watch.ElapsedMilliseconds;

            return regions;
        }
    }

    //public partial class VideoSurveilance : Form1
    //{
    //    //declaring global variables
    //    private static Capture _capture;
    //    //takes images from camera as image frames
    //    public VideoSurveilance()
    //    {
    //        InitializeComponent();
    //        _capture = new Emgu.CV.Capture();
    //        _capture.Start();
    //        ProcessFrame();
    //        // Run()
    ////    }

    //    private void ProcessFrame2()
    //    {

    //        Mat frame = _capture.QueryFrame();
    //        Image<Bgr, Byte> ImageFrame = frame.ToImage<Bgr, Byte>();

    //        if (ImageFrame != null)   // confirm that image is valid
    //        {
    //            Image<Gray, byte> grayframe = ImageFrame.Convert<Gray, byte>();

    //            long processingTime;
    //            Rectangle[] results = FindPedestrian.Find(frame, false, out processingTime);
    //            foreach (Rectangle rect in results)
    //                ImageFrame.Draw(rect, new Bgr(Color.Red), 10);
    //        }
    //        imageBox1.Image = ImageFrame;
    //        //imageBox2.Image = frame;
    //    }



}
