#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <signal.h>
#include <string>
#include <chrono>
#include <thread>

#include "Lepton3.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

// ----> Global variables
Lepton3* lepton3=nullptr;
static bool close = false;
// <---- Global variables

// ----> Global functions
void close_handler(int s);
// <---- Global functions

int main (int argc, char *argv[])
{
    cout << "Lepton thermal data grabber" << std::endl;

    // SIGINT handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = close_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    Lepton3::DebugLvl deb_lvl = Lepton3::DBG_NONE;

    lepton3 = new Lepton3("/dev/spidev0.0", "/dev/i2c-0", deb_lvl); // use SPI1 and I2C-1 ports
    lepton3->start();

    uint64_t frameIdx=0;
    uint16_t min;
    uint16_t max;
    uint8_t w,h;

    if (lepton3->enableRadiometry(true) != LEP_OK)
    {
        cerr << "Failed to enable radiometry!" << std::endl;
        return EXIT_FAILURE;
    }

    if (lepton3->setGainMode(LEP_SYS_GAIN_MODE_HIGH) == LEP_OK)
    {

        LEP_SYS_GAIN_MODE_E gainMode;
        if(lepton3->getGainMode(gainMode) == LEP_OK)
        {
	    string str = (gainMode==LEP_SYS_GAIN_MODE_HIGH)?string("High"):((gainMode==LEP_SYS_GAIN_MODE_LOW)?string("Low"):string("Auto"));
	    cout << " * Gain mode: " << str << std::endl;
        }
    }

    if (lepton3->enableAgc(false) != LEP_OK)
    {
        cerr << "Failed to set agc status!" << std::endl;
    }

    if (lepton3->doFFC() == LEP_OK)
    {
        cout << " * FFC completed" << std::endl;
    }

    while (!close)
    {
        const uint16_t* data16 = lepton3->getLastFrame16(w, h, &min, &max);

        if (data16)
        {
            cv::Mat frame16(h, w, CV_16UC1);

            if (data16)
            {
                memcpy(frame16.data, data16, w*h*sizeof(uint16_t));
                cout << frame16 << std::endl;
            }

            if (++frameIdx == 200){
                frameIdx = 0;

                if (lepton3->doFFC() == LEP_OK)
                {
                    cout << " * FFC completed" << std::endl;
                }

                if (lepton3->enableRadiometry(true) != LEP_OK)
                {
                    cerr << "Failed to enable radiometry!" << std::endl;
                }

                if (lepton3->setGainMode(LEP_SYS_GAIN_MODE_HIGH) == LEP_OK)
                {
                    LEP_SYS_GAIN_MODE_E gainMode;
                    if (lepton3->getGainMode(gainMode) == LEP_OK)
                    {
                    string str = (gainMode==LEP_SYS_GAIN_MODE_HIGH)?string("High"):((gainMode==LEP_SYS_GAIN_MODE_LOW)?string("Low"):string("Auto"));
                    cout << " * Gain mode: " << str << std::endl;
                    }
                }

                if (lepton3->enableAgc(false) != LEP_OK)
                {
                    cerr << "Failed to set agc status!" << std::endl;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    delete lepton3;

    return EXIT_SUCCESS;
}

void close_handler(int s)
{
    if(s==2)
    {
        cout << std::endl << "Closing thermal grabber..." << std::endl;
        close = true;
    }
}