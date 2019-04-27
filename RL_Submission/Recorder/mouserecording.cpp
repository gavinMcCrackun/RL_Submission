#include "stdafx.h"
#include <Windows.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

using namespace std;

ofstream outfile;
std::chrono::time_point<std::chrono::steady_clock> start;
HWND window;

bool file_exists(std::string filename)
{
	ifstream infile(filename.c_str());
	if (!infile.is_open())
		return false;

	infile.close();
	return true;
}

class MyHook {
public:
	//singleton
	static MyHook& Instance() {
		static MyHook myHook;
		return myHook;
	}
	HHOOK hook; // handle to the hook

	void InstallHook(); // function to install our hook
	void UninstallHook(); // function to uninstall our hook

	MSG msg; // struct with information about all messages in our queue
	int Messsages(); // function to "deal" with our messages 
};
LRESULT WINAPI MyMouseCallback(int nCode, WPARAM wParam, LPARAM lParam); //callback declaration

int MyHook::Messsages() {
	while (msg.message != WM_QUIT) { //while we do not close our application
		if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		Sleep(1);
	}
	UninstallHook(); //if we close, let's uninstall our hook
	return (int)msg.wParam; //return the messages
}

void MyHook::InstallHook() {
	/*
	SetWindowHookEx(
	WM_MOUSE_LL = mouse low level hook type,
	MyMouseCallback = our callback function that will deal with system messages about mouse
	NULL, 0);

	c++ note: we can check the return SetWindowsHookEx like this because:
	If it return NULL, a NULL value is 0 and 0 is false.
	*/
	if (!(hook = SetWindowsHookEx(WH_MOUSE_LL, MyMouseCallback, NULL, 0))) {
		printf_s("Error: %x \n", GetLastError());
	}
}

void MyHook::UninstallHook() {
	/*
	uninstall our hook using the hook handle
	*/
	UnhookWindowsHookEx(hook);
}

void write_window_pos()
{
	LPCWSTR name = L"OSBuddy Guest - Guest";
	window = FindWindow(NULL, name);

	RECT rect;
	GetWindowRect(window, &rect);

	outfile << "(" << rect.left << ", " << rect.top << ")  "
		<< "(" << rect.right << ", " << rect.bottom << ")"
		<< endl;
}

LRESULT WINAPI MyMouseCallback(int nCode, WPARAM wParam, LPARAM lParam) {
	MSLLHOOKSTRUCT * pMouseStruct = (MSLLHOOKSTRUCT *)lParam; // WH_MOUSE_LL struct
															  /*
															  nCode, this parameters will determine how to process a message
															  This callback in this case only have information when it is 0 (HC_ACTION): wParam and lParam contain info

															  wParam is about WINDOWS MESSAGE, in this case MOUSE messages.
															  lParam is information contained in the structure MSLLHOOKSTRUCT
															  */

	if (nCode == 0) { // we have information in wParam/lParam ? If yes, let's check it:

		chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
		outfile << elapsed.count() << ' ';
		POINT pt;
		GetCursorPos(&pt);
		ScreenToClient(window, &pt);
		if (pMouseStruct != NULL)
		{
			// Mouse struct contain information?			
			outfile << pt.x << " " << pt.y;
		}

		switch (wParam)
		{
		case WM_LBUTTONUP:
			outfile << " LEFT_UP";
			break;
		case WM_LBUTTONDOWN:
			outfile << " LEFT_DOWN";
			break;
		case WM_RBUTTONUP:
			outfile << " RIGHT_UP";
			break;
		case WM_RBUTTONDOWN:
			outfile << " RIGHT_DOWN";
			break;
		}

		outfile << endl;
	}

	/*
	Every time that the nCode is less than 0 we need to CallNextHookEx:
	-> Pass to the next hook
	MSDN: Calling CallNextHookEx is optional, but it is highly recommended;
	otherwise, other applications that have installed hooks will not receive hook notifications and may behave incorrectly as a result.
	*/
	return CallNextHookEx(MyHook::Instance().hook, nCode, wParam, lParam);
}

string next_filename()
{
	for (int i = 0; i < 1e5; ++i)
	{
		stringstream numstring;
		numstring << setw(5) << setfill('0') << i;

		string filename = "data" + numstring.str() + ".txt";
		if (!file_exists(filename))
			return filename;
	}
	return " ";
}

int main() {
	start = std::chrono::steady_clock::now();
	outfile.open(next_filename());
	outfile.precision(15);
	write_window_pos();

	MyHook::Instance().InstallHook();
	int ret = MyHook::Instance().Messsages();

	outfile.close();
	return ret;
}