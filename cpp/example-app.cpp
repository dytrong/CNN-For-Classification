#include <torch/script.h> // One-stop header.
#include<opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <time.h>

using namespace std;

using namespace cv;

torch::Tensor change_image_to_tensor(string image_path) {

    auto image = cv::imread(image_path, cv::ImreadModes::IMREAD_COLOR);

    cv::Mat image_transfomed;

    cv::resize(image, image_transfomed, cv::Size(224, 224));

    cv::cvtColor(image_transfomed, image_transfomed, cv::COLOR_BGR2RGB);

    // 图像转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,3},torch::kByte);

    tensor_image = tensor_image.permute({2,0,1});

    tensor_image = tensor_image.toType(torch::kFloat);

    tensor_image = tensor_image.div(255);

    tensor_image = tensor_image.unsqueeze(0);

    return tensor_image;
}

void show_detect_info(int prediction)
{
  switch(prediction)
  {
      case 0: 
          cout<<"OK--光亮"<<endl;
          break;
      case 1:
          cout<<"OK--阴影"<<endl;
          break;
      case 2:
          cout<<"Not OK--带阴影的黑块"<<endl;
          break;
      case 3:
          cout<<"Not OK--黑块"<<endl;
          break;
      default:
          cout<<"输入值不符合"<<endl;

  }
}

int main(int argc, const char* argv[]) {

  if (argc != 2) {

    std::cerr << "usage: example-app <path-to-exported-script-module>\n";

    return -1;
  }

  // Deserialize the ScriptModule from a file using torch::jit::load().
  std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

  //端面的图片地址
  string image_path = "/home/dytrong/drawCirclePics识别标识/Pics20190524截取结果/face/端面识别/imgs/10.jpg";
  
  clock_t start, finish;

  double totaltime;

  start = clock();
  
  torch::Tensor inputs = change_image_to_tensor(image_path);

  // Execute the model and turn its output into a tensor.
  auto output = module->forward({inputs}).toTensor();;
  
  auto max_result = output.max(1,true);
  
  //预测断面情况
  //0表示ok_光亮,1表示OK_阴影,2表示带阴影的黑块,3表示只有黑块 
  auto prediction = std::get<1>(max_result).item<float>();
  
  //输出信息
  show_detect_info(prediction);

  finish = clock();
  
  totaltime = (double)(finish - start)/CLOCKS_PER_SEC;

  cout<<"判断断面运行时间为 "<<totaltime<<" 秒!"<<endl;
 
}
