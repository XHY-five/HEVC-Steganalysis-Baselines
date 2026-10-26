/*
 * TU-only research hook for x265 4.1.
 * This file is a new source file, distributed with the TU-only patch.
 * Read X265_TU_PLAN, produce X265_TU_FINAL and X265_TU_REPORT.
 * x265's original source remains under its original GPL licensing.
 */
#ifndef X265_TU_STEGO_BRIDGE_H
#define X265_TU_STEGO_BRIDGE_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <sstream>
#include <vector>
#include <fstream>

struct XingTUKey
{
    int frame, ctu, x, y;
    XingTUKey(int f=0, int c=0, int a=0, int b=0) : frame(f), ctu(c), x(a), y(b) {}
    bool operator<(const XingTUKey& b) const
    {
        if (frame != b.frame) return frame < b.frame;
        if (ctu != b.ctu) return ctu < b.ctu;
        if (y != b.y) return y < b.y;
        return x < b.x;
    }
};

struct XingTURDCost
{
    double full, split;
    int ipm;
    XingTURDCost(double f=0.0, double s=0.0, int mode=0)
        : full(f), split(s), ipm(mode) {}
};

class XingTUBridge
{
    std::mutex lock_;
    std::map<XingTUKey, int> plan_;
    std::map<XingTUKey, XingTURDCost> rd_;
    std::map<XingTUKey, int> actual_;
    FILE* final_;
    std::string report_;
    bool active_;

    static std::vector<std::string> cols(const std::string& line)
    {
        std::vector<std::string> r;
        std::istringstream ss(line);
        std::string part;
        while (std::getline(ss, part, ',')) r.push_back(part);
        return r;
    }
    static int columnIndex(const std::vector<std::string>& v, const char* name)
    {
        for (unsigned i=0; i<v.size(); ++i)
            if (v[i] == name) return (int)i;
        return -1;
    }
    static std::string columnValue(const std::vector<std::string>& row, int n)
    {
        return n>=0 && (size_t)n<row.size() ? row[(size_t)n] : std::string();
    }
    void loadPlan(const char* filename)
    {
        std::ifstream file(filename);
        if (!file.good())
        {
            fprintf(stderr, "[xing-tu] ERROR unable to open X265_TU_PLAN=%s\n", filename);
            std::abort();
        }
        std::string line;
        if (!std::getline(file, line))
        {
            fprintf(stderr, "[xing-tu] ERROR empty plan\n");
            std::abort();
        }
        std::vector<std::string> h=cols(line);
        const int frame=columnIndex(h,"frame"), ctu=columnIndex(h,"ctu"), x=columnIndex(h,"x"), y=columnIndex(h,"y");
        const int target=columnIndex(h,"target_bit"), used=columnIndex(h,"used");
        if (frame<0 || ctu<0 || x<0 || y<0 || target<0)
        {
            fprintf(stderr, "[xing-tu] ERROR invalid TU plan columns\n");
            std::abort();
        }
        int readRows=0;
        while (std::getline(file,line))
        {
            if (line.empty()) continue;
            std::vector<std::string> r=cols(line);
            if (used>=0 && std::atoi(columnValue(r,used).c_str()) != 1) continue;
            int bit=std::atoi(columnValue(r,target).c_str());
            if (bit!=0 && bit!=1) { fprintf(stderr,"[xing-tu] ERROR bad target bit\n"); std::abort(); }
            XingTUKey k(std::atoi(columnValue(r,frame).c_str()),
                        std::atoi(columnValue(r,ctu).c_str()),
                        std::atoi(columnValue(r,x).c_str()),
                        std::atoi(columnValue(r,y).c_str()));
            if (!plan_.insert(std::make_pair(k,bit)).second)
            {
                fprintf(stderr, "[xing-tu] ERROR duplicated TU plan carrier\n");
                std::abort();
            }
            ++readRows;
        }
        fprintf(stderr, "[xing-tu] Loaded %d TU decisions from %s\n", readRows, filename);
    }

    XingTUBridge() : final_(NULL), active_(false)
    {
        const char* plan=std::getenv("X265_TU_PLAN");
        const char* output=std::getenv("X265_TU_FINAL");
        const char* report=std::getenv("X265_TU_REPORT");
        if (plan && *plan) { loadPlan(plan); active_=true; }
        if (output && *output)
        {
            final_=std::fopen(output,"w");
            if (!final_) { fprintf(stderr,"[xing-tu] ERROR cannot create %s\n",output); std::abort(); }
            std::fprintf(final_, "frame,ctu,x,y,cu_size,pu_mode,pred_mode,tu_bit,j_single,j_split,n4x4\n");
            active_=true;
        }
        if (report && *report) { report_=report; active_=true; }
    }
    ~XingTUBridge()
    {
        if (final_) std::fclose(final_);
        if (report_.empty()) return;
        FILE* f=std::fopen(report_.c_str(),"w");
        if (!f) return;
        std::fprintf(f,"frame,ctu,x,y,target,actual,status\n");
        for (std::map<XingTUKey,int>::const_iterator it=plan_.begin();it!=plan_.end();++it)
        {
            std::map<XingTUKey,int>::const_iterator got=actual_.find(it->first);
            const char* status=(got==actual_.end() ? "MISSING" : (it->second==got->second ? "PASS" : "MISMATCH"));
            std::fprintf(f,"%d,%d,%d,%d,%d,",it->first.frame,it->first.ctu,it->first.x,it->first.y,it->second);
            if (got!=actual_.end()) std::fprintf(f,"%d",got->second);
            std::fprintf(f,",%s\n",status);
        }
        std::fclose(f);
    }
    XingTUBridge(const XingTUBridge&);
    XingTUBridge& operator=(const XingTUBridge&);
public:
    static XingTUBridge& instance() { static XingTUBridge b; return b; }
    bool active() const { return active_; }
    int target(int frame,int ctu,int x,int y)
    {
        if (plan_.empty()) return -1;
        std::lock_guard<std::mutex> guard(lock_);
        std::map<XingTUKey,int>::const_iterator it=plan_.find(XingTUKey(frame,ctu,x,y));
        return it==plan_.end() ? -1 : it->second;
    }
    void observe(int frame,int ctu,int x,int y,int ipm,double full,double split)
    {
        if (!active_) return;
        if (!std::isfinite(full) || !std::isfinite(split)) return;
        std::lock_guard<std::mutex> guard(lock_);
        rd_[XingTUKey(frame,ctu,x,y)]=XingTURDCost(full,split,ipm);
    }
    void recordFinal(int frame,int ctu,int x,int y,int tuBit,int n4x4)
    {
        if (!active_) return;
        XingTUKey key(frame,ctu,x,y);
        std::lock_guard<std::mutex> guard(lock_);
        actual_[key]=tuBit;
        if (final_)
        {
            std::map<XingTUKey,XingTURDCost>::const_iterator r=rd_.find(key);
            std::fprintf(final_,"%d,%d,%d,%d,8x8,2Nx2N,intra,%d,",frame,ctu,x,y,tuBit);
            if (r!=rd_.end())
                std::fprintf(final_,"%.9g,%.9g",r->second.full,r->second.split);
            else std::fprintf(final_,",");
            std::fprintf(final_,",%d\n",n4x4);
            std::fflush(final_);
        }
    }
};
#endif
