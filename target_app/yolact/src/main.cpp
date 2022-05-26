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
#include "unistd.h"

// Header files for OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Header files for Vitis AI
#include <yolact.hpp>
#include <lnx_time.hpp>

// Namespaces
using namespace std;
using namespace cv;

void print_usage()
{
  cout << "Usage: ./yolact.exe --image <image-file.jpg> [options]" << endl;
  cout << endl;
  cout << "  options:" << endl;

  cout << "  --help" << endl;
  cout << "      Prints this menu" << endl;

  cout << "  --iter N" << endl;
  cout << "      Performs processing over N iterations for average single-threaded performance measurement" << endl;

  cout << "  --no_display" << endl;
  cout << "      Turns off display output of processed images" << endl;

  cout << "  --score_threshold N" << endl;
  cout << "      Removes detections that fall below the provided N threshold" << endl;

  cout << "  --verbose or -v" << endl;
  cout << "      Prints status & performance information" << endl;
  cout << endl;
}

/*
 * @brief main - Main entry point of application.
 *
 */
int main( int argc, char *argv[] )
{
  lnx_timer init_timer;
  lnx_timer run_timer;

  vector<string> img_files;
  int img_provided = 0;
  float score_thresh = 0.0f;
  int iter = 0;
  int verbose = 0;
  int display = 1;

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
        img_provided = 1;
        iter++;
        i += 2;
      }
      else if (!strcmp(argv[i], "--score_threshold"))
      {
        score_thresh = atof(argv[i+1]);
        i += 2;
      }
      else if (!strcmp(argv[i], "--iter"))
      {
        iter = atoi(argv[i+1]);
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
      else
      {
        cout << "ERROR: input argument " << argv[i] << " not recognized." << endl;
        print_usage();
        return -1;
      }
    }
  }

  if (img_provided != 1)
  {
    cout << "ERROR: please provide input image as argument" << endl;
    print_usage();
    return -1;
  }

  init_timer.reset();
  run_timer.reset();

  init_timer.start();
  yolact yolact_model;
  yolact_model.create("model/yolact.xmodel");
  init_timer.stop();

  if (verbose)
  {
    cout << "Input files:" << endl;
    for (auto &img_file : img_files)
    {
      cout << img_file << endl;
    }
    cout << "score threshold set to " << score_thresh << endl;
  }

  /* Read frame from file */
  if (verbose) cout << "Reading image" << endl;
  vector<cv::Mat> frames;
  vector<cv::Mat> results;
  for (auto &img_file : img_files)
  {
    cv::Mat frame = imread(img_file);

    if (frame.empty())
    {
      cout << "ERROR: input file " << img_file << " is empty" << endl;
      return -1;
    }

    frames.push_back(frame);
  }

  /* Run the model */
  if (verbose || iter > 1) cout << "Testing model for " << iter << " iterations" << endl;
  int pct_done = 0;

  for (int i = 0; i < iter; i++)
  {
    cv::Mat result;
    run_timer.start();
    yolact_model.run( frames[i%frames.size()], result, score_thresh );
    run_timer.stop();
    cv::putText(result, "RUN #" + to_string(i+1), cv::Point(5,result.rows-10), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255,0,0), 1, cv::LINE_AA, 0);
    results.push_back(result);

    if (iter > 1)
    {
      if (i % (int)(0.1*iter) == 0)
      {
        printf("\r%d%% complete", pct_done);
        fflush(stdout);
        pct_done += 10;
      }
    }
  }

  if (iter > 1)
  {
    printf("\r100%% complete\n");
    fflush(stdout);
  }

  if (verbose || iter > 1)
  {
    cout << "DISCLAIMER: This application is not optimized for performance." << endl;
    cout << "  Only a single software thread is used, and DPU batch capabilites" << endl;
    cout << "  are not currently supported." << endl;
    cout << endl;

    char time_str[20];
    sprintf(time_str, "%1.3f", init_timer.avg_secs());
    cout << "Initialization time took " << time_str << endl;
    sprintf(time_str, "%1.3f", run_timer.avg_secs());
    cout << "Run time took " << time_str << " per frame using a single thread" << endl;

    yolact_model.print_stats();
    cout << endl;
  }

  if (display)
  {
    cout << "Displaying results for 5 seconds ... hit the any key to close the current display" << endl;
    for (auto &result : results)
    {
      cv::imshow("Result", result);
      cv::waitKey(5000);
    }
  }

  return 0;
}

