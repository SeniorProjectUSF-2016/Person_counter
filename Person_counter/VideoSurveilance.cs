//----------------------------------------------------------------------------
//  Copyright (C) 2004-2016 by EMGU Corporation. All rights reserved.       
//----------------------------------------------------------------------------

using System.Threading.Tasks;
using System.IO;
using System.IO.Ports;


using System.Runtime.InteropServices;
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
using Emgu.CV.XFeatures2D;
using Emgu.CV.VideoSurveillance;
using System.Diagnostics;
using Emgu.CV.Util;
using static Emgu.CV.FileNode;
#if !(__IOS__ || NETFX_CORE)
using Emgu.CV.Cuda;
#endif
using Emgu.CV.Features2D;


namespace VideoSurveilance
{
    public partial class VideoSurveilance : Form
    {
        //Rectangle rect;
        //private static MCvFont _font = new MCvFont(Emgu.CV.CvEnum.FontType.HersheyPlain, 1.0, 1.0);
        private static Capture _cameraCapture;

        private Rectangle[] arr = new Rectangle[10000];
        long processingTime;
        List<Mat> people = new List<Mat>();
      

        private int counter = 0;

        private static BackgroundSubtractor _fgDetector;
        private static Emgu.CV.Cvb.CvBlobDetector _blobDetector;
        public static String GetTimestamp(DateTime value)
        {
            return value.ToString("yyyyMMddHHmmssffff");
        }
        String timeStamp = GetTimestamp(DateTime.Now);

        //public Rectangle this[int i]
        //{
        //    get
        //    {
        //        // This indexer is very simple, and just returns or sets
        //        // the corresponding element from the internal array.
        //        return arr[i];
        //    }
        //    set
        //    {
        //        arr[i] = value;
        //    }
        //}

        public Mat this[int i]
        {
            get
            {
                // This indexer is very simple, and just returns or sets
                // the corresponding element from the internal array.
                return people[i];
            }
            set
            {
                people[i] = value;
            }
        }

        public VideoSurveilance()
        {
            //System.ComponentModel.IContainer components = new System.ComponentModel.Container();
            //SerialPort serialPort1 = new SerialPort("COM3");
            //serialPort1.BaudRate = 9600;
            //serialPort1.DtrEnable = true;
            //if (!serialPort1.IsOpen)
            //{
            //    try
            //    {
            //        serialPort1.Open();
            //        serialPort1.Write("T");
            //        serialPort1.Close();
            //    }
            //    catch
            //    {
            //        MessageBox.Show("There was an error. Please make sure that the correct port was selected, and the device, plugged in.");
            //    }
            //}
            //serialPort1.Open();


            //string sensor = serialPort1.ReadLine();

            //string posResult = "true";

            //if (String.Compare(sensor, posResult, true) == 1)
            //{
                InitializeComponent();
                Run();

            //}

        }

        void Run()
        {
            try
            {
                // test for test
                _cameraCapture = new Capture(2);
                //_cameraCapture.FlipHorizontal = !_cameraCapture.FlipHorizontal;

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
            long matchTime;
            Mat result = new Mat();
            Mat mask = new Mat();
            Mat homography = new Mat();


            Rectangle personROI = new Rectangle();

            Mat modelImage = new Mat();
            Mat observedImage = new Mat();



            Rectangle roi = new Rectangle(213, 0, 213, 480); // set the roi image.ROI = new Rectangle(x, Y, Width, Height);
            //ImageFrame.ROI = roi;

            ImageFrame.Draw(roi, new Bgr(Color.Green), 5);

            if (ImageFrame != null)   // confirm that image is valid
            {

                Rectangle[] results = FindPedestrian.Find(frame, true, out processingTime);
                if (people.Count == 0)
                {
                    foreach (Rectangle rect in results)
                    {

                        if (rect.Width >= 150)
                        {
                            personROI.X = rect.X;
                            personROI.Y = rect.Y;
                            personROI.Width = rect.Width;
                            personROI.Height = rect.Height;

                            Mat person = new Mat(frame, personROI);
                            people.Add(person);

                            ImageFrame.Draw(rect, new Bgr(Color.Red), 5);
                            //Console.WriteLine(index);
                        }
                    }

                }
                else
                {
                    foreach (Rectangle rect in results)
                    {
                        ImageFrame.Draw(rect, new Bgr(Color.Red), 5);
                        var check = false;
                        var temp = new List<Mat>(people);

                        foreach (Mat aperson in people)
                        {
                            Mat img = new Mat(frame, rect);
                            observedImage = aperson;
                            modelImage = img;
                            result = Draw(modelImage, observedImage, out matchTime, out check);
                            if (!check)
                            {
                                temp.Add(img);
                                ++counter;
                                Console.WriteLine("Counter: " + counter);
                                break;
                            }

                        }
                        people = new List<Mat>(temp);

                        Console.WriteLine("End for frame processing");
                    }
                }

                Mat smoothedFrame = new Mat();
                    CvInvoke.GaussianBlur(frame, smoothedFrame, new Size(3, 3), 1); //filter out noises
                                                                                    //frame._SmoothGaussian(3); 

                    #region use the BG/FG detector to find the forground mask
                    Mat forgroundMask = new Mat();
                    //_fgDetector.Apply(smoothedFrame, forgroundMask);

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
                    imageBox2.Image = result;
                    //imageBox2.Image = forgroundMask;



                }
            //people.Clear();
            }




        public bool FindMatch(Mat modelImage, Mat observedImage, out long matchTime, out VectorOfKeyPoint modelKeyPoints, out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {
            int newimage = 0;
            newimage++;
            int k = 2;
            double uniquenessThreshold = 0.8;
            double hessianThresh = 300;

            Stopwatch watch;
            homography = null;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();

#if !__IOS__
            if (CudaInvoke.HasCuda)
            {
                CudaSURF surfCuda = new CudaSURF((float)hessianThresh);
                using (GpuMat gpuModelImage = new GpuMat(modelImage))
                //extract features from the object image
                using (GpuMat gpuModelKeyPoints = surfCuda.DetectKeyPointsRaw(gpuModelImage, null))
                using (GpuMat gpuModelDescriptors = surfCuda.ComputeDescriptorsRaw(gpuModelImage, null, gpuModelKeyPoints))
                using (CudaBFMatcher matcher = new CudaBFMatcher(DistanceType.L2))
                {
                    surfCuda.DownloadKeypoints(gpuModelKeyPoints, modelKeyPoints);
                    watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    using (GpuMat gpuObservedImage = new GpuMat(observedImage))
                    using (GpuMat gpuObservedKeyPoints = surfCuda.DetectKeyPointsRaw(gpuObservedImage, null))
                    using (GpuMat gpuObservedDescriptors = surfCuda.ComputeDescriptorsRaw(gpuObservedImage, null, gpuObservedKeyPoints))
                    //using (GpuMat tmp = new GpuMat())
                    //using (Stream stream = new Stream())
                    {
                        matcher.KnnMatch(gpuObservedDescriptors, gpuModelDescriptors, matches, k);

                        surfCuda.DownloadKeypoints(gpuObservedKeyPoints, observedKeyPoints);

                        mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                        mask.SetTo(new MCvScalar(255));
                        Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                        int nonZeroCount = CvInvoke.CountNonZero(mask);
                        if (nonZeroCount >= 4)
                        {
                            nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                               matches, mask, 1.5, 20);
                            if (nonZeroCount >= 4)
                                homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                                   observedKeyPoints, matches, mask, 2);
                        }
                    }
                    watch.Stop();
                }
            }
            else
#endif
            {
                using (UMat uModelImage = modelImage.ToUMat(AccessType.Read))
                using (UMat uObservedImage = observedImage.ToUMat(AccessType.Read))
                {
                    SURF surfCPU = new SURF(hessianThresh);
                    //extract features from the object image
                    UMat modelDescriptors = new UMat();
                    surfCPU.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);

                    watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    UMat observedDescriptors = new UMat();
                    surfCPU.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
                    BFMatcher matcher = new BFMatcher(DistanceType.L2);
                    matcher.Add(modelDescriptors);

                    matcher.KnnMatch(observedDescriptors, matches, k, null);
                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                           matches, mask, 1.5, 20);
                        Console.WriteLine("Match Points: " + nonZeroCount);

                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                               observedKeyPoints, matches, mask, 2);

                        if (nonZeroCount > 15)
                        {
                            watch.Stop();
                            matchTime = watch.ElapsedMilliseconds;
                            return true;
                        }
                    }

                    watch.Stop();
                }
            }
            matchTime = watch.ElapsedMilliseconds;
            return false;
        }

        /// <summary>
        /// Draw the model image and observed image, the matched features and homography projection.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The model image and observed image, the matched features and homography projection.</returns>
        public Mat Draw(Mat modelImage, Mat observedImage, out long matchTime, out bool check)
        {
            int newimage = 0;

            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask = new Mat();
                check = FindMatch(modelImage, observedImage, out matchTime, out modelKeyPoints, out observedKeyPoints, matches,
                   out mask, out homography);
               // Console.WriteLine("Model points: " + modelKeyPoints.Size + "  Observed points:" + observedKeyPoints.Size + "  Match points:"+ matches.Size);
                //Draw the matched keypoints
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                   matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);

                #region draw the projected region on the image

                if (homography != null)
                {
                    //draw a rectangle along the projected model
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                    using (VectorOfPoint vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                    }
                    

                }

                #endregion

                return result;

            }
        }
        }



    public static class DetectFace
        {
            public static void Detect(
              Mat image, String faceFileName, String eyeFileName,
              List<Rectangle> faces, List<Rectangle> eyes,
              bool tryUseCuda,
              out long detectionTime)
            {
                Stopwatch watch;
            
                {
                    //Read the HaarCascade objects
                    using (CascadeClassifier face = new CascadeClassifier(faceFileName))
                    using (CascadeClassifier eye = new CascadeClassifier(eyeFileName))
                    {
                        watch = Stopwatch.StartNew();
                        using (UMat ugray = new UMat())
                        {
                            CvInvoke.CvtColor(image, ugray, Emgu.CV.CvEnum.ColorConversion.Bgr2Gray);

                            //normalizes brightness and increases contrast of the image
                            CvInvoke.EqualizeHist(ugray, ugray);

                            //Detect the faces  from the gray scale image and store the locations as rectangle
                            //The first dimensional is the channel
                            //The second dimension is the index of the rectangle in the specific channel
                            Rectangle[] facesDetected = face.DetectMultiScale(
                               ugray,
                               1.1,
                               10,
                               new Size(20, 20));

                        
                            faces.AddRange(facesDetected);

                            foreach (Rectangle f in facesDetected)
                            {
                                //Get the region of interest on the faces
                                using (UMat faceRegion = new UMat(ugray, f))
                                {
                                    Rectangle[] eyesDetected = eye.DetectMultiScale(
                                       faceRegion,
                                       1.1,
                                       10,
                                       new Size(20, 20));

                                    foreach (Rectangle e in eyesDetected)
                                    {
                                        Rectangle eyeRect = e;
                                        eyeRect.Offset(f.X, f.Y);
                                        eyes.Add(eyeRect);
                                    }
                                }
                            }
                        }
                        watch.Stop();
                    }
                }
                detectionTime = watch.ElapsedMilliseconds;
            }
        }
    

    public static class FindPedestrian
        {

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

                        MCvObjectDetection[] results = des.DetectMultiScale(umat,0, default(Size), default(Size), 1.15,2,false);
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
    }