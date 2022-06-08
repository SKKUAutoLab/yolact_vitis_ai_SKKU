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
#include "unistd.h"

// Header files for OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Header files for Vitis AI
#include <vitis/ai/graph_runner.hpp>
#include <vitis/ai/nnpp/apply_nms.hpp>

// Timer class
#include "lnx_time.hpp"
#include "coco_labels.hpp"

// Model constants
#define PROTO_HW    (138)
#define PROTO_C     (32)
#define PROTO_SIZE  (PROTO_HW*PROTO_HW*PROTO_C)
#define NUM_PRIORS  (19248)

// COCO dataset classes
#define NUM_CLASSES (81)

// Detection constants
#define NMS_CONF_THRESH (0.05f)
#define NMS_THRESH      (0.5f)
#define NMS_TOP_K       (200)
#define KEEP_TOP_K      (5)

// Overlay constants
#define MASK_ALPHA (0.45f)

// DEBUG
//#define SHOW_PROTO_IMAGES 1

class yolact
{
  public:

    yolact()
    {
      pre_timer.reset();
      exec_timer.reset();
      post_timer.reset();
      overlay_timer.reset();
    }

    ~yolact()
    {
      free(proto_data);
      free(loc_data);
      free(conf_data);
      free(mask_data);
      free(prior_data);
    }

    int create( std::string xmodel )
    {
      /* Create the graph runner */
      graph  = xir::Graph::deserialize(xmodel);
      attr   = xir::Attrs::create();
      runner = vitis::ai::GraphRunner::create_graph_runner(graph.get(), attr.get());
      CHECK(runner != nullptr);

      /* Determine batch size */
      auto input_tensor_buffer = runner->get_inputs();
      batch_size = input_tensor_buffer[0]->get_tensor()->get_shape().at(0);

      /* Allocate prototype output buffers */
      proto_data = (float *)malloc(sizeof(float)*PROTO_SIZE*batch_size); // 138x138x32

      /* Allocate location data output buffer */
      loc_data = (float *)malloc(sizeof(float)*NUM_PRIORS*4*batch_size);

      /* Allocate confidence data output buffers */
      conf_data = (float *)malloc(sizeof(float)*NUM_PRIORS*NUM_CLASSES*batch_size);

      /* Allocate mask data output buffers */
      mask_data = (float *)malloc(sizeof(float)*NUM_PRIORS*PROTO_C*batch_size);

      /* Compute prior boxes */
      prior_data = (box_t *)malloc(sizeof(box_t)*NUM_PRIORS*batch_size);
      create_priors(prior_data);

      return batch_size;
    }

    void run( std::vector<cv::Mat> &img,
              float                 nms_conf_thresh,
              float                 nms_thresh,
              float                 score_thresh )
    {
      /* Save threshold values */
      l_nms_thresh = (nms_thresh < 0.0f) ? NMS_THRESH : nms_thresh;
      l_nms_conf_thresh = (nms_conf_thresh < 0.0f) ? NMS_CONF_THRESH : nms_conf_thresh;

      /* Get the input/output tensor buffer handles */
      auto l_runner = runner.get();
      auto in_tensor_buff = l_runner->get_inputs();
      auto out_tensor_buff = l_runner->get_outputs();

      /* Process input data */
      int iter = 0;
      while (iter < img.size())
      {
        std::vector<cv::Mat> img_buff;
        for (int b = 0; b < batch_size; b++)
        {
          img_buff.push_back(img[iter+b]);
        }

        /* Pre-process the data */
        pre_timer.start();
        preprocess(img_buff, in_tensor_buff);

        /* Sync input tensor buffers */
        for (auto& input : in_tensor_buff)
        {
          input->sync_for_write(0, input->get_tensor()->get_data_size() / input->get_tensor()->get_shape()[0]);
        }
        pre_timer.stop();

        /* Execute the graph */
        exec_timer.start();
        auto v = l_runner->execute_async(in_tensor_buff, out_tensor_buff);
        auto status = l_runner->wait((int)v.first, -1);
        CHECK_EQ(status, 0) << "failed to run the graph";
        exec_timer.stop();

        /* Sync output tensor buffers */
        post_timer.start();
        for (auto output : out_tensor_buff)
        {
          output->sync_for_read(0, output->get_tensor()->get_data_size() / output->get_tensor()->get_shape()[0]);
        }

        /* Post-process that data */
        postprocess(out_tensor_buff);
        post_timer.stop();

        /* Create graphic overlays */
        overlay_timer.start();
        create_overlays(img_buff, score_thresh);

        for (int b = 0; b < batch_size; b++)
        {
          img[iter+b] = img_buff[b];
        }
        overlay_timer.stop();

        iter += batch_size;
      }
    }

    void print_stats( )
    {
      char time_str[20];
      sprintf(time_str, "%1.3f", pre_timer.avg_secs() / (float)batch_size);
      std::cout << "Average pre-processing  time (CPU)       = " << time_str << " seconds" << std::endl;
      sprintf(time_str, "%1.3f", exec_timer.avg_secs() / (float)batch_size);
      std::cout << "Average graph execution time (CPU + DPU) = " << time_str << " seconds" << std::endl;
      sprintf(time_str, "%1.3f", post_timer.avg_secs() / (float)batch_size);
      std::cout << "Average post-processing time (CPU)       = " << time_str << " seconds" << std::endl;
      sprintf(time_str, "%1.3f", overlay_timer.avg_secs() / (float)batch_size);
      std::cout << "Average graphic overlay time (CPU)       = " << time_str << " seconds" << std::endl;
    }

  private:

    /*************************************************************************
     *  Data types                                                           *
     *************************************************************************/
    typedef struct
    {
      int   label;
      float score;
      float x;
      float y;
      float w;
      float h;
    } box_t;

    /*************************************************************************
     * Local variables & constants                                           *
     *************************************************************************/
    std::unique_ptr<xir::Graph> graph;
    std::unique_ptr<xir::Attrs> attr;
    std::unique_ptr<vart::RunnerExt> runner;
    float *loc_data;
    float *conf_data;
    float *mask_data;
    float *proto_data;
    box_t *prior_data;
    std::map<int, std::vector<float>> decoded_bboxes;
    std::map<int, std::vector<float>> masks;
    std::vector<box_t> box_results;
    std::vector<std::vector<float>> mask_results;
    std::vector<int> batch_index;
    int batch_size;
    float l_nms_conf_thresh;
    float l_nms_thresh;

    lnx_timer pre_timer, exec_timer, post_timer, overlay_timer;

    /*************************************************************************
     * Functions                                                             *
     *************************************************************************/

    /* This function taken from
     * Vitis-AI/demo/Vitis-AI-Library/samples/graph_runner/resnet50_graph_runner/resnet50_graph_runner.cpp
     */
    int get_fix_point(const xir::Tensor* tensor)
    {
      CHECK(tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << tensor->get_name();
      return tensor->template get_attr<int>("fix_point");
    }

    /* This function taken from
     * Vitis-AI/demo/Vitis-AI-Library/samples/graph_runner/resnet50_graph_runner/resnet50_graph_runner.cpp
     */
    std::vector<std::int32_t> get_index_zeros(const xir::Tensor* tensor)
    {
      auto ret = tensor->get_shape();
      std::fill(ret.begin(), ret.end(), 0);
      return ret;
    }

    /* Create prior boxes */
    void create_priors(box_t *prior_data)
    {
      /* The following configuration is used to create priors (based on yolact/data/config.py):
       *   backbone.use_pixel_scales = True
       *   backbone.use_square_anchors = True
       *   backbone.preapply_sqrt = True
       */
      const float inv_max_size = 1.0f / 550.0f;  // Maximum image size (550x550)
      const int num_priors = 19248;
      const int fmap_dims[5] = {69, 35, 18, 9, 5};
      const int scales[5] = {24, 48, 96, 192, 384};
      const float aspect_ratios[5][3] = { {1.0f, 0.7071068f, 1.4142135f},
                                          {1.0f, 0.7071068f, 1.4142135f},
                                          {1.0f, 0.7071068f, 1.4142135f},
                                          {1.0f, 0.7071068f, 1.4142135f},
                                          {1.0f, 0.7071068f, 1.4142135f} };

      box_t prior_box;

      for (int k = 0; k < 5; k++)
      {
        float scale = scales[k];
        float inv_fmap_dims = 1.0f / (float)fmap_dims[k];

        for (int j = 0; j < fmap_dims[k]; j++)
        {
          for (int i = 0; i < fmap_dims[k]; i++)
          {
            prior_box.x = ((float)i + 0.5f) * inv_fmap_dims;
            prior_box.y = ((float)j + 0.5f) * inv_fmap_dims;

            for (int r = 0; r < 3; r++)
            {
              prior_box.w = scale * aspect_ratios[k][r] * inv_max_size;
              prior_box.h = prior_box.w;
              *prior_data++ = prior_box;
            }
          }
        }
      }

    }

    /* This function modified from
     * Vitis-AI/demo/Vitis-AI-Library/samples/graph_runner/resnet50_graph_runner/resnet50_graph_runner.cpp
     */
    void preprocess( std::vector<cv::Mat>             &img,
                     std::vector<vart::TensorBuffer*> &input_tensor_buffers)
    {
      auto input_tensor = input_tensor_buffers[0]->get_tensor();
      auto batch = input_tensor->get_shape().at(0);
      auto height = input_tensor->get_shape().at(1);
      auto width = input_tensor->get_shape().at(2);

      int fixpos = get_fix_point(input_tensor);
      float input_fixed_scale = std::exp2f(1.0f * (float)fixpos);

      auto size = cv::Size(width, height);

      for (int index = 0; index < batch; ++index)
      {
        cv::Mat resize_image;
        cv::Mat frame = img[index];
        if (size != frame.size())
        {
          cv::resize(frame, resize_image, size);
        }
        else
        {
          frame.copyTo(resize_image);
        }

        uint64_t data_in = 0u;
        size_t size_in = 0u;
        auto idx = get_index_zeros(input_tensor);
        idx[0] = (int)index;
        std::tie(data_in, size_in) = input_tensor_buffers[0]->data(idx);
        set_input_image(resize_image, (void*)data_in, input_fixed_scale);
      }
    }

    /* This function modified from
     * Vitis-AI/demo/Vitis-AI-Library/samples/graph_runner/resnet50_graph_runner/resnet50_graph_runner.cpp
     * The mean & scale values are taken from yolact/data/config.py and are slightly different
     * than expected for the torchvision pre-trained imagenet ResNet-50 model.  Torchvision models use mean
     * values of [103.53, 116.28, 123.675] and std (i.e. scale) values of [57.375, 57.12, 58.395].
     */
    static void set_input_image(const cv::Mat& image, void* data_in, float fix_scale)
    {
      float mean[3] = {103.94f, 116.78f, 123.68f}; // BGR
      float scale[3] = {fix_scale/57.38f, fix_scale/57.12f, fix_scale/58.4f};
      signed char *data = (signed char*)data_in;

      for (int h = 0; h < image.rows; h++)
      {
        for (int w = 0; w < image.cols; w++)
        {
          for (int c = 0; c < 3; c++)
          {
            auto image_data = ((float)image.at<cv::Vec3b>(h, w)[c] - mean[c]) * scale[c];
            data[h * image.cols * 3 + w * 3 + c] = (int)(image_data + 0.5f);
          }
        }
      }
    }

    /* Debug function to show prototype images */
    void show_prototypes(float *proto_data)
    {
      cv::Mat proto_img[PROTO_C];
      float max_vals[PROTO_C] = {0};

      for (int c = 0; c < PROTO_C; c++)
      {
        proto_img[c].create(cv::Size(PROTO_HW, PROTO_HW), CV_32FC1);
      }

      for (int h = 0; h < PROTO_HW; h++)
      {
        for (int w = 0; w < PROTO_HW; w++)
        {
          for (int c = 0; c < PROTO_C; c++)
          {
            proto_img[c].at<float>(h,w) = proto_data[h*PROTO_HW*PROTO_C + w*PROTO_C + c];

            if (proto_img[c].at<float>(h,w) > max_vals[c]) max_vals[c] = proto_img[c].at<float>(h,w);
          }
        }
      }

      for (int c = 0; c < PROTO_C; c++)
      {
        cv::Mat color_img;
        color_img = proto_img[c]/max_vals[c]*255;
        color_img.convertTo(color_img, CV_8UC1);
        cv::applyColorMap(color_img, color_img, cv::COLORMAP_JET);
        cv::resize(color_img, color_img, cv::Size(550, 550));
        cv::imshow("ProtoType", color_img);
        cv::waitKey(100);
      }
    }

    /* Debug function to dump prototype data to csv file */
    void dump_prototypes( float *proto_data )
    {
      FILE *proto_file[PROTO_C];

      for (int c = 0; c < PROTO_C; c++)
      {
        std::string proto_str = "proto_data_" + std::to_string(c) + ".csv";
        proto_file[c] = fopen(proto_str.c_str(), "w");
      }

      for (int h = 0; h < PROTO_HW; h++)
      {
        for (int w = 0; w < PROTO_HW; w++)
        {
          for (int c = 0; c < PROTO_C; c++)
          {
            fprintf(proto_file[c], "%f", proto_data[h*PROTO_HW*PROTO_C + w*PROTO_C + c]);
            if (w < PROTO_HW-1)
            {
              fprintf(proto_file[c], ", ");
            }
            else
            {
              fprintf(proto_file[c], "\n");
            }
          }
        }
      }

      for (int c = 0; c < PROTO_C; c++)
      {
        fclose(proto_file[c]);
      }
    }

    /* Sigmoid function */
    float sigmoid( float x )
    {
      return 1.0f / (1.0f + exp(-x));
    }

    /* Sort the final results by score */
    void sort_results(  std::vector<box_t>              &boxes,
                        std::vector<std::vector<float>> &masks,
                        int                              batch_start,
                        int                              batch_end )
    {
      for (int i = batch_start; i < batch_end; i++)
      {
        for (int k = i+1; k < (batch_end-batch_start); k++)
        {
          if (boxes[k].score > boxes[i].score)
          {
            box_t temp_box = boxes[i];
            boxes[i] = boxes[k];
            boxes[k] = temp_box;

            std::vector<float> temp_mask = masks[i];
            masks[i] = masks[k];
            masks[k] = temp_mask;
          }
        }
      }

    }

    /* Adds mask overlays to output image */
    void draw_masks( cv::Mat                         &img,
                     std::vector<box_t>               boxes,
                     std::vector<std::vector<float>>  masks,
                     int                              batch_start,
                     int                              batch_end,
                     float                           *proto_data,
                     float                            score_thresh )
    {
      int c_idx = 0;

      for (int i = batch_start; i < batch_end; i++)
      {
        if (boxes[i].score < score_thresh)
        {
          continue;
        }

        auto mask = masks[i];
        cv::Mat m1(cv::Size(PROTO_HW, PROTO_HW), CV_32FC1);

        /* Compute m1 = sigmoid(proto * mask') */
        for (int h = 0; h < PROTO_HW; h++)
        {
          for (int w = 0; w < PROTO_HW; w++)
          {
            float sum = 0.0f;
            for (int c = 0; c < PROTO_C; c++)
            {
              sum += (proto_data[h*PROTO_HW*PROTO_C + w*PROTO_C + c] * mask[c]);
            }

            m1.at<float>(h,w) = sigmoid(sum);
          }
        }

        /* Resize the mask to the image dimensions */
        cv::Mat m2;
        cv::resize(m1, m2, img.size());

        /* Crop the mask to within the bounding-box region */
        cv::Rect roi;
        float width  = img.cols;
        float height = img.rows;
        roi.x        = std::min(std::max(boxes[i].x * width, 0.0f), width);
        roi.y        = std::min(std::max(boxes[i].y * height, 0.0f), height);
        roi.width    = std::min(std::max(boxes[i].w * width, 0.0f), width);
        roi.height   = std::min(std::max(boxes[i].h * height, 0.0f), height);
        cv::Mat crop(cv::Size(roi.width, roi.height), m2.type());
        m2(roi).copyTo(crop);
        m2 = cv::Mat::zeros(m2.size(), m2.type());
        crop.copyTo(m2(roi));

        /* Apply mask to input image mask_img = (img * mask_alpha) + () mask_color * (1 - mask_alpha)) */
        cv::Scalar color = get_color(c_idx++);

        for (int h = 0; h < m2.rows; h++)
        {
          for (int w = 0; w < m2.cols; w++)
          {
            if (m2.at<float>(h,w) > 0.5f)
            {
              for (int c = 0; c < 3; c++)
              {
                img.at<cv::Vec3b>(h,w)[c] = img.at<cv::Vec3b>(h,w)[c] * MASK_ALPHA + color[c] * (1.0f - MASK_ALPHA);
              }
            }
          }
        }

      }
    }

    /* Adds bounding boxes to output image */
    void draw_boxes( cv::Mat &img, std::vector<box_t> boxes, int batch_start, int batch_end, float score_thresh )
    {
      float width = img.cols;
      float height = img.rows;
      int c_idx = 0;

      for (int i = batch_start; i < batch_end; i++)
      {
        box_t box = boxes[i];
        if (box.score < score_thresh)
        {
          continue;
        }

        /* Compute x-y coordinates relative to the input image size */
        int xmin = std::min(std::max(box.x * width, 0.0f), width);
        int ymin = std::min(std::max(box.y * height, 0.0f), height);
        int xmax = std::min(std::max(xmin + (box.w * width), 0.0f), width);
        int ymax = std::min(std::max(ymin + (box.h * height), 0.0f), height);

        /* Get the bounding box color & draw the bounding box on the input image */
        cv::Scalar color = get_color(c_idx++);
        cv::rectangle(img, cv::Point(xmin, ymin), cv::Point(xmax, ymax), color, 1, 1, 0);

        /* Format the score & class label text */
        char score_str[4];
        sprintf(score_str, "%1.2f", box.score);
        std::string label = coco_labels[box.label] + ": " + std::string(score_str);
        cv::Size txt_size = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, 0.6, 1, NULL);

        /* Draw the class label & score on the image */
        cv::Rect roi;
        roi.x = xmin;
        roi.y = std::max(ymin-txt_size.height-8, 0);
        roi.width = (roi.x + txt_size.width+2) > width ? (width - roi.x) : txt_size.width+2;
        roi.height = (roi.y + txt_size.height+8) > height ? (height - roi.y) : txt_size.height+8;
        img(roi) = color;
        cv::putText(img, label, cv::Point(roi.x, roi.y+txt_size.height), cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(255,255,255), 1, cv::LINE_AA, 0);
      }
    }

    // This function modified from Vitis-AI/tools/Vitis-AI-Library/xnnpp/src/ssd/ssd_detector.cpp
    void decode_bbox( float *bbox_ptr,
                      int    idx )
    {
      const float var[2] = {0.1f, 0.2f};
      std::vector<float> bbox(4);

      for (int i = 0; i < 4; i++)
      {
        bbox[i] = bbox_ptr[i];
      }

      box_t prior_box = prior_data[idx];

      float decode_bbox_center_x, decode_bbox_center_y;
      float decode_bbox_width, decode_bbox_height;

      // Compute center-point & width/height
      decode_bbox_center_x = prior_box.x + bbox[0] * var[0] * prior_box.w;
      decode_bbox_center_y = prior_box.y + bbox[1] * var[0] * prior_box.h;
      decode_bbox_width    = prior_box.w * exp(bbox[2] * var[1]);
      decode_bbox_height   = prior_box.h * exp(bbox[3] * var[1]);

      // x-y bounds
      bbox[0] = decode_bbox_center_x - decode_bbox_width  / 2; // x-min
      bbox[1] = decode_bbox_center_y - decode_bbox_height / 2; // y-min
      bbox[2] = decode_bbox_center_x + decode_bbox_width  / 2; // x-max
      bbox[3] = decode_bbox_center_y + decode_bbox_height / 2; // y-max

      // bbox x,y,w,h
      bbox[0] = std::max(std::min(bbox[0], 1.f), 0.f);
      bbox[1] = std::max(std::min(bbox[1], 1.f), 0.f);
      bbox[2] = std::max(std::min(bbox[2], 1.f), 0.f);
      bbox[3] = std::max(std::min(bbox[3], 1.f), 0.f);

      // Convert to center coordinates
      bbox[0] = 0.5f * (bbox[0] + bbox[2]); // x-center
      bbox[1] = 0.5f * (bbox[1] + bbox[3]); // y-center
      bbox[2] = (bbox[2] - bbox[0]) * 2.0f; // width
      bbox[3] = (bbox[3] - bbox[1]) * 2.0f; // height

      decoded_bboxes.emplace(idx, std::move(bbox));
    }

    // This function modified from Vitis-AI/tools/Vitis-AI-Library/xnnpp/src/ssd/ssd_detector.cpp
    void get_one_class_max_score_index( const float              *conf_data,
                                        int                       label,
                                        vector<pair<float, int>> *score_index_vec)
    {
      conf_data += label;
      for (int i = 0; i < NUM_PRIORS; i++)
      {
        auto score = *conf_data;

        if (score > l_nms_conf_thresh)
        {
          score_index_vec->emplace_back(score, i);
        }
        conf_data += NUM_CLASSES;
      }

      std::stable_sort(
          score_index_vec->begin(), score_index_vec->end(),
          [](const pair<float, int>& lhs, const pair<float, int>& rhs) {
            return lhs.first > rhs.first;
          });

      if (NMS_TOP_K < score_index_vec->size())
      {
        score_index_vec->resize(NMS_TOP_K);
      }
    }

    // This function modified from Vitis-AI/tools/Vitis-AI-Library/xnnpp/src/ssd/ssd_detector.cpp
    void get_multi_class_max_score_index( const float                      *conf_data,
                                          int                               start_label,
                                          int                               num_classes,
                                          vector<vector<pair<float, int>>> *score_index_vec)
    {
      for (auto i = start_label; i < start_label + num_classes; i++)
      {
        get_one_class_max_score_index(conf_data, i, &((*score_index_vec)[i]));
      }
    }

    // This function modified from Vitis-AI/tools/Vitis-AI-Library/xnnpp/src/ssd/ssd_detector.cpp
    void apply_one_class_nms( float                           *loc_data,
                              float                           *mask_data,
                              int                              label,
                              std::vector<pair<float, int>>   &score_index_vec,
                              std::vector<int>                *indices )
    {
      std::vector<size_t>        results;
      std::vector<vector<float>> boxes;
      std::vector<float>         mask;
      std::vector<float>         scores;
      std::map<size_t, int>      resultmap;

      indices->clear();
      int i = 0;

      while (i < score_index_vec.size())
      {
        uint32_t idx = score_index_vec[i].second;

        if (decoded_bboxes.find(idx) == decoded_bboxes.end()) // decoded bbox not found
        {
          if ( idx < NUM_PRIORS )
          {
            decode_bbox( &loc_data[idx*4], idx );

            for (int c = 0; c < PROTO_C; c++)
            {
              mask.push_back(mask_data[idx*PROTO_C+c]);
            }
            masks.emplace( idx, std::move(mask) );

          }
        }

        boxes.push_back(decoded_bboxes[idx]);
        scores.push_back(score_index_vec[i].first);
        resultmap[i] = idx;
        i++;
      }

      applyNMS( boxes, scores, l_nms_thresh, l_nms_conf_thresh, results );

      for (auto &r : results)
      {
        indices->push_back(resultmap[r]);
      }
    }

    // This function modified from Vitis-AI/tools/Vitis-AI-Library/xnnpp/src/ssd/ssd_detector.cpp
    void detect( float                           *loc_data,
                 float                           *conf_data,
                 float                           *mask_data,
                 box_t                           *prior_data,
                 float                           *proto_data,
                 std::vector<box_t>               &box_result,
                 std::vector<std::vector<float>>  &mask_result,
                 std::vector<int>                 &batch_index )
    {

      decoded_bboxes.clear();
      masks.clear();

      int num_det = 0;
      vector<vector<int>> indices(NUM_CLASSES);
      vector<vector<pair<float, int>>> score_index_vec(NUM_CLASSES);

      // Get top_k scores (with corresponding indices).
      get_multi_class_max_score_index(conf_data, 1, NUM_CLASSES-1, &score_index_vec);

      // Skip the background class by starting at 1 instead of 0
      for (int c = 1; c < NUM_CLASSES; c++)
      {
        // Perform NMS for one class
        apply_one_class_nms( loc_data, mask_data, c, score_index_vec[c], &(indices[c]) );
        num_det += indices[c].size();
      }

      if (KEEP_TOP_K > 0 && num_det > KEEP_TOP_K)
      {
        vector<tuple<float, int, int>> score_index_tuples;
        for (auto label = 0u; label < NUM_CLASSES; ++label)
        {
          const vector<int>& label_indices = indices[label];
          for (auto j = 0u; j < label_indices.size(); ++j)
          {
            auto idx = label_indices[j];
            auto score = conf_data[idx * NUM_CLASSES + label];
            score_index_tuples.emplace_back(score, label, idx);
          }
        }

        // Keep top k results per image.
        std::sort(score_index_tuples.begin(), score_index_tuples.end(),
                  [](const tuple<float, int, int>& lhs,
                     const tuple<float, int, int>& rhs) {
                    return get<0>(lhs) > get<0>(rhs);
                  });

        score_index_tuples.resize(KEEP_TOP_K);

        indices.clear();
        indices.resize(NUM_CLASSES);

        for (auto& item : score_index_tuples)
        {
          indices[get<1>(item)].push_back(get<2>(item));
        }
      }

      int b_idx = 0;
      if (batch_index.empty())
      {
        batch_index.push_back(b_idx);
      }
      else
      {
        b_idx = batch_index.back();
      }

      for (auto label = 1u; label < indices.size(); ++label)
      {
        for (auto idx : indices[label])
        {
          auto score = conf_data[idx * NUM_CLASSES + label];
          auto& bbox = decoded_bboxes[idx];
          auto& mask = masks[idx];
          box_t box_res;
          box_res.label = label;
          box_res.score = score;
          box_res.x = bbox[0] - 0.5f * bbox[2];
          box_res.y = bbox[1] - 0.5f * bbox[3];
          box_res.w = bbox[2];
          box_res.h = bbox[3];
          box_result.emplace_back(box_res);
          mask_result.emplace_back(mask);
          b_idx++;
        }
      }
      batch_index.push_back(b_idx);
    }

    /* Copies data from tensor output buffer to host memory */
    void copy_data( float *input,
                    float *output,
                    size_t size,
                    int    batch,
                    int    elements,
                    int    channels )
    {
      int offset = 0;
      for (int i = 0; i < batch; i++)
      {
        memcpy( &output[offset], &input[i*elements], size );
        offset += (NUM_PRIORS * channels);
      }
    }

    void postprocess( const std::vector<vart::TensorBuffer*>& output_tensor_buffer )
    {
      uint64_t data_out = 0;
      size_t size_out = 0;
      int batch = 1;

      /* Copy tensor output data to host memory */
      for (auto &tensor_buffer : output_tensor_buffer)
      {
        auto output_tensor = tensor_buffer->get_tensor();
        std::string tensor_name = output_tensor->get_name();
        auto idx = get_index_zeros(output_tensor);
        idx[0] = 0;
        std::tie(data_out, size_out) = tensor_buffer->data(idx);
        auto shape = output_tensor->get_shape();
        batch = shape.front();
        int num_elements = output_tensor->get_element_num() / batch;
        int num_channels = shape.back();
        size_out /= batch;

        /* Prototype output */
        if (!tensor_name.compare("Yolact__Yolact_13022_fix_")) // Prototype output
        {
          memcpy(proto_data, (float *)data_out, size_out*batch);

#ifdef SHOW_PROTO_IMAGES
          show_prototypes(proto_data);
#endif

#ifdef DUMP_PROTO_DATA
          dump_prototypes(proto_data);
#endif
        }

        /* Copy mask data to host memory */
        else if (!tensor_name.compare("Yolact__Yolact_13715"))
        {
          copy_data( (float *)data_out, mask_data, size_out, batch, num_elements, num_channels );
        }

        /* Copy confidence data to host memory */
        else if (!tensor_name.compare("Yolact__Yolact_13718"))
        {
          copy_data( (float *)data_out, conf_data, size_out, batch, num_elements, num_channels );
        }

        /* Copy location data to host memory */
        else if (!tensor_name.compare("Yolact__Yolact_13708_fix_"))
        {
          copy_data( (float *)data_out, loc_data, size_out, batch, num_elements, num_channels );
        }
      }

      /* Process detections */
      box_results.clear();
      mask_results.clear();
      batch_index.clear();

      for (int b = 0; b < batch; b++)
      {
        detect( &loc_data[NUM_PRIORS*4*b],
                &conf_data[NUM_PRIORS*NUM_CLASSES*b],
                &mask_data[NUM_PRIORS*PROTO_C*b],
                &prior_data[NUM_PRIORS*4*b],
                &proto_data[PROTO_SIZE*b],
                 box_results,
                 mask_results,
                 batch_index );
      }
    }

    /* Mask & box color look-up */
    cv::Scalar get_color(int label)
    {
      cv::Scalar colors[19] =
      {
        cv::Scalar(54,67,244),  cv::Scalar(99,30,233),   cv::Scalar(176,39,156), cv::Scalar(183,58,103),
        cv::Scalar(181,81,63),  cv::Scalar(243,150,33),  cv::Scalar(244,169,3),  cv::Scalar(212,188,0),
        cv::Scalar(136,150,0),  cv::Scalar(80,175,76),   cv::Scalar(74,195,139), cv::Scalar(57,220,205),
        cv::Scalar(59,235,255), cv::Scalar(7,193,255),   cv::Scalar(0,152,255),  cv::Scalar(34,87,255),
        cv::Scalar(72,85,72),   cv::Scalar(158,158,158), cv::Scalar(139,125,96)
      };

      return colors[(label*5)%19];
    }

    void create_overlays( std::vector<cv::Mat> &img, float score_thresh )
    {
      /* Draw output image overlay */
      for (int i = 0; i < img.size(); i++)
      {
        int batch_start = batch_index[i];
        int batch_end   = batch_index[i+1];

        // Sort the results based on score so colors look the same as running the model on dev. machine
        sort_results(box_results, mask_results, batch_start, batch_end);

        draw_masks( img[i], box_results, mask_results, batch_start, batch_end, &proto_data[PROTO_SIZE*i], score_thresh );
        draw_boxes( img[i], box_results, batch_start, batch_end, score_thresh );
      }
    }

};
