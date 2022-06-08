/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <thread>
#include <unistd.h>

// Header files for OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Application Header files
#include "yolact.hpp"
#include "lnx_time.hpp"

// Namespaces
using namespace std;

/*
 * Help usage
 */
void print_usage()
{
  cout << "Usage: ./yolact.exe --image <image-file.jpg> [options]" << endl;
  cout << endl;
  cout << "  options:" << endl;

  cout << "  --help" << endl;
  cout << "      Prints this menu" << endl;

  cout << "  --iter N" << endl;
  cout << "      Performs processing over N iterations for performance measurement" << endl;
  cout << "      Setting this to a value greater than 1 will turn off visulization of the outputs" << endl;

  cout << "  --no_display" << endl;
  cout << "      Turns off display output of processed images" << endl;

  cout << "  --score_thresh N" << endl;
  cout << "      Removes detections post NMS processing that fall below the provided N threshold (default = 0.0)" << endl;

  cout << "  --nms_conf_thresh N" << endl;
  cout << "      Removes detections prior to NMS processing that fall below the provided N threshold (default = 0.05)" << endl;

  cout << "  --nms_thresh N" << endl;
  cout << "      NMS IoU N threshold (default = 0.5)" << endl;

  cout << "  --threads N" << endl;
  cout << "      Specifies the number of thread to use for processing (default = 1)" << endl;

  cout << "  --wait N" << endl;
  cout << "      Specifies the wait time in seconds between output image displays (default = 5 seconds)" << endl;

  cout << "  --verbose or -v" << endl;
  cout << "      Prints status & performance information" << endl;
  cout << endl;
}

/*
 * Main entry point of application.
 *
 */
int main( int argc, char *argv[] )
{
  lnx_timer init_timer;
  lnx_timer run_timer;
  vector<string> img_files;
  float score_thresh = 0.0f;
  float nms_thresh = -1.0f;
  float nms_conf_thresh = -1.0f;
  int iter = 1;
  int test_iter = 0;
  int img_cnt = 0;
  int verbose = 0;
  int display = 1;
  int num_threads = 1;
  int disp_wait = 5000;

  /* Process input arguments */
  {
    int i = 1;
    while(i < argc)
    {
      if (!strcmp(argv[i], "--image"))
      {
        if ( i+1 >= argc )
        {
          cout << "ERROR: please provide input image as argument" << endl;
          print_usage();
          return -1;
        }

        if (!std::filesystem::exists(argv[i+1]))
        {
          cout << "ERROR: provided image file " << argv[i+1] << " does not exist." << endl;
          cout << "       Please provide a valid image file" << endl;
          return -1;
        }

        img_files.push_back(string(argv[i+1]));
        img_cnt++;
        i += 2;
      }
      else if (!strcmp(argv[i], "--score_thresh"))
      {
        score_thresh = atof(argv[i+1]);
        i += 2;
      }
      else if (!strcmp(argv[i], "--nms_conf_thresh"))
      {
        nms_conf_thresh = atof(argv[i+1]);
        i += 2;
      }
      else if (!strcmp(argv[i], "--nms_thresh"))
      {
        nms_thresh = atof(argv[i+1]);
        i += 2;
      }
      else if (!strcmp(argv[i], "--iter"))
      {
        test_iter = atoi(argv[i+1]);
        display=0;
        i += 2;
      }
      else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose"))
      {
        verbose = 1;
        i++;
      }
      else if (!strcmp(argv[i], "--help"))
      {
        print_usage();
        i++;
      }
      else if (!strcmp(argv[i], "--no_display"))
      {
        display = 0;
        i++;
      }
      else if (!strcmp(argv[i], "--wait"))
      {
        disp_wait = (int)(atof(argv[i+1]) * 1000);
        i+=2;
      }
      else if (!strcmp(argv[i], "--threads"))
      {
        num_threads = atoi(argv[i+1]);
        i+=2;
      }
      else
      {
        cout << "ERROR: input argument " << argv[i] << " not recognized." << endl;
        print_usage();
        return -1;
      }
    }
  }
  cout << endl;

  if (img_cnt < 1)
  {
    cout << "ERROR: please provide input image as argument" << endl;
    print_usage();
    return -1;
  }

  auto nproc = std::thread::hardware_concurrency();
  if (num_threads > nproc)
  {
    cout << "WARNING: You are using " << num_threads
         << " threads, but your machine only has " << nproc << " processors." << endl;
    cout << " Consider lowering the thread-count to " << nproc << " to prevent lock-ups" << endl << endl;
  }

  /* Display input parameters */
  if (verbose)
  {
    cout << "Input files:" << endl;
    for (auto &img_file : img_files)
    {
      cout << img_file << endl;
    }

    cout << endl;
    cout << "Input image count:        " << img_files.size() << endl;
    cout << "Score threshold:          " << score_thresh << endl;
    cout << "NMS confidence threshold: " << ((nms_conf_thresh < 0) ? NMS_CONF_THRESH : nms_conf_thresh) << endl;
    cout << "NMS IoU threshold:        " << ((nms_thresh < 0) ? NMS_THRESH : nms_thresh) << endl;
    cout << "Display output:           " << ((display == 1) ? "ON" : "OFF") << endl;
    cout << "Test iterations:          " << test_iter << endl;
    cout << "Processing threads:       " << num_threads << endl;
    cout << endl;
  }

  /* Reset run timers */
  init_timer.reset();
  run_timer.reset();

  /* Model initialization */
  init_timer.start();

  yolact yolact_model[num_threads];
  int batch_size = yolact_model[0].create("model/yolact.xmodel");

  for (int i = 1; i < num_threads; i++)
  {
    yolact_model[i].create("model/yolact.xmodel");
  }

  init_timer.stop();

  /* Read frame from file */
  if (verbose) cout << "Reading image" << endl;

  vector<cv::Mat> frames;
  for (auto &img_file : img_files)
  {
    cv::Mat frame = cv::imread(img_file);

    if (frame.empty())
    {
      cout << "ERROR: input file " << img_file << " is empty" << endl;
      return -1;
    }

    frames.push_back(frame);
  }

  /* Compute how many iterations to process based on user input & DPU batch capabilites */
  if (test_iter > 0)
  {
    iter = (test_iter + batch_size * num_threads - 1) / (batch_size * num_threads);
  }
  else
  {
    iter = (img_cnt + batch_size * num_threads - 1) / (batch_size * num_threads);
  }

  /* Run the model */
  if (verbose || test_iter > 0) cout << "Testing model";
  if (test_iter > 0)
  {
    frames.resize(1);  // only use the first input image for performance testing
    cout << " for " << test_iter << ((test_iter > 1) ? " iterations" : " iteration");
  }

  cout << endl;

  /* Allocatin and load memory for input/output buffers */
  vector<vector<cv::Mat>> images(num_threads);

  for (int i = 0; i < iter; i++)
  {
    for (int t = 0; t < num_threads; t++)
    {
      for (int b = 0; b < batch_size; b++)
      {
        int img_idx = (i*batch_size*num_threads + t*batch_size + b) % frames.size();
        cv::Mat temp;
        frames[img_idx].copyTo(temp);
        images[t].push_back(temp);
      }
    }
  }
  frames.clear();

  /* Spawn processing threads */
  run_timer.start();
  std::vector<thread> threads(num_threads);

  for (int t = 0; t < num_threads; t++)
  {
    threads[t] = thread( &yolact::run,
                         &yolact_model[t],
                          std::ref(images[t]),
                          nms_conf_thresh,
                          nms_thresh,
                          score_thresh
                       );
  }

  /* Wait for thread completions */
  for (int t = 0; t < num_threads; t++)
  {
    threads[t].join();
  }

  run_timer.stop();

  /* Display timing results */
  if (verbose || test_iter > 0)
  {
    char time_str[20];
    char fps_str[20];
    sprintf(time_str, "%1.3f", init_timer.avg_secs());
    cout << "Initialization time took " << time_str << endl;
    float avg_proc_time = run_timer.avg_secs() / (float)(batch_size * num_threads * iter);
    sprintf(time_str, "%1.3f", avg_proc_time);
    sprintf(fps_str, "%.1f", 1.0f / avg_proc_time);
    cout << "Average run time was " << time_str << " seconds/frame (FPS = " << fps_str << ") using " << num_threads
         << ((num_threads == 1) ? " thread" : " threads") << endl;

    if (verbose)
    {
      for (int t = 0; t < num_threads; t++)
      {
        cout << "Thread " << t << ":" << endl;
        yolact_model[t].print_stats();
      }
    }

    cout << endl;
  }

  /* Display processed images */
  if (display)
  {
    if (disp_wait == 0)
    {
      cout << "Displaying results ... hit any key to advance to the next output image" << endl;
    }
    else
    {
      cout << "Displaying results for " << (float)disp_wait/1000 << " seconds ... hit any key to close the current display" << endl;
    }

    for (int i = 0; i < iter; i++)
    {
      for (int t = 0; t < num_threads; t++)
      {
        for (int b = 0; b < batch_size; b++)
        {
          if (i*num_threads*batch_size + t*batch_size + b >= img_cnt) break;
          auto result = images[t][i*batch_size+b];
          cv::imshow("Result", result);
          cv::waitKey(disp_wait);
        }
      }
    }
  }

  cout << "Done." << endl;
  return 0;
}

