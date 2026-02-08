#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdint>
#include <fstream>
#include <unordered_set>

using namespace std;

class RandomStreamGen {
public:
    RandomStreamGen(size_t n, uint64_t seed) : n_(n), rng_(seed) {}

    vector<string> generateStream() {
        vector<string> s;
        s.reserve(n_);
        for (size_t i = 0; i < n_; ++i) s.push_back(genOne());
        return s;
    }

    vector<size_t> splitPrefixesByPercent(int stepPercent) const {
        if (stepPercent <= 0 || stepPercent > 100) {
            throw invalid_argument("stepPercent must be in [1..100]");
        }
        vector<size_t> pref;
        for (int p = stepPercent; p <= 100; p += stepPercent) {
            size_t k = (size_t)((__int128)n_ * p / 100);
            if (k == 0) k = 1;
            if (k > n_) k = n_;
            if (pref.empty() || pref.back() != k) pref.push_back(k);
        }
        if (pref.empty() || pref.back() != n_) pref.push_back(n_);
        return pref;
    }

private:
    size_t n_;
    mt19937_64 rng_;
    static constexpr const char* ALPHABET =
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789-";

    char randChar() {
        static constexpr int ALPH_LEN = 63;
        uniform_int_distribution<int> d(0, ALPH_LEN - 1);
        return ALPHABET[d(rng_)];
    }

    string genOne() {
        uniform_int_distribution<int> lenDist(1, 30);
        int L = lenDist(rng_);
        string x;
        x.resize(L);
        for (int i = 0; i < L; ++i) x[i] = randChar();
        return x;
    }
};

static inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

static inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

static uint32_t murmur3_32(const uint8_t* data, size_t len, uint32_t seed) {
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    uint32_t h1 = seed;
    const int nblocks = (int)(len / 4);

    const uint32_t* blocks = (const uint32_t*)(data);
    for (int i = 0; i < nblocks; i++) {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= (uint32_t)tail[2] << 16;
        case 2: k1 ^= (uint32_t)tail[1] << 8;
        case 1: k1 ^= (uint32_t)tail[0];
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }

    h1 ^= (uint32_t)len;
    h1 = fmix32(h1);
    return h1;
}

class HashFuncGen {
public:
    explicit HashFuncGen(uint32_t seed) : seed_(seed) {}
    uint32_t operator()(const string& s) const {
        return murmur3_32(reinterpret_cast<const uint8_t*>(s.data()), s.size(), seed_);
    }
private:
    uint32_t seed_;
};

class HyperLogLog {
public:
    explicit HyperLogLog(int B) : B_(B), m_(1u << B), regs_(m_, 0) {
        if (B_ < 4 || B_ > 16) {
            throw invalid_argument("B must be in [4..16]");
        }
    }

    void reset() {
        fill(regs_.begin(), regs_.end(), 0);
    }

    void add(uint32_t x) {
        uint32_t idx = x >> (32 - B_);
        uint32_t w = (x << B_);
        int lz = __builtin_clz(w);
        int rho = lz + 1;
        uint8_t &r = regs_[idx];
        if (rho > r) r = (uint8_t)rho;
    }

    double estimate() const {
        double Z = 0.0;
        int V = 0;
        for (uint32_t i = 0; i < m_; ++i) {
            Z += ldexp(1.0, -(int)regs_[i]);
            if (regs_[i] == 0) V++;
        }

        double E = alpha_m() * (double)m_ * (double)m_ / Z;

        if (E <= 2.5 * (double)m_ && V > 0) {
            return (double)m_ * log((double)m_ / (double)V);
        }

        return E;
    }

    uint32_t m() const { return m_; }

private:
    int B_;
    uint32_t m_;
    vector<uint8_t> regs_;

    double alpha_m() const {
        if (m_ == 16) return 0.673;
        if (m_ == 32) return 0.697;
        if (m_ == 64) return 0.709;
        return 0.7213 / (1.0 + 1.079 / (double)m_);
    }
};

static size_t exactDistinctPrefix(const vector<string>& stream, size_t pref) {
    unordered_set<string> st;
    st.reserve(pref * 2);
    for (size_t i = 0; i < pref; ++i) st.insert(stream[i]);
    return st.size();
}

int main(int argc, char** argv) {
    int streams = 20;
    size_t n = 200000;
    int stepPercent = 5;
    int B = 12;
    uint64_t baseSeed = 42;
    string outCsv = "results.csv";

    RandomStreamGen tmp(n, baseSeed);
    vector<size_t> prefixes = tmp.splitPrefixesByPercent(stepPercent);
    int T = (int)prefixes.size();

    vector<vector<double>> estByStep(T), exactByStep(T);
    for (int t = 0; t < T; ++t) {
        estByStep[t].reserve(streams);
        exactByStep[t].reserve(streams);
    }

    for (int s = 0; s < streams; ++s) {
        uint64_t streamSeed = baseSeed + 1000003ULL * (uint64_t)s;
        RandomStreamGen gen(n, streamSeed);
        auto stream = gen.generateStream();
        HashFuncGen hf((uint32_t)(1234567u + 104729u * (uint32_t)s));
        HyperLogLog hll(B);

        size_t pos = 0;
        for (int t = 0; t < T; ++t) {
            size_t needPref = prefixes[t];

            while (pos < needPref) {
                hll.add(hf(stream[pos]));
                pos++;
            }

            double est = hll.estimate();
            double ex = (double)exactDistinctPrefix(stream, needPref);

            estByStep[t].push_back(est);
            exactByStep[t].push_back(ex);
        }
    }

    ofstream fout(outCsv);
    fout << "step,processed,mean_exact,mean_est,std_est\n";

    for (int t = 0; t < T; ++t) {
        double sumE = 0, sumX = 0;
        for (auto e : estByStep[t]) sumE += e;
        for (auto x : exactByStep[t]) sumX += x;
        double meanE = sumE / streams;
        double meanX = sumX / streams;

        double var = 0;
        for (auto e : estByStep[t]) var += (e - meanE) * (e - meanE);
        double stdE = sqrt(var / (streams - 1));

        fout << (t + 1) << "," << prefixes[t] << "," << meanX << "," << meanE << "," << stdE << "\n";
    }

    cerr << "Wrote: " << outCsv << "\n";
    cerr << "B=" << B << " m=" << (1u << B) << "\n";
    return 0;
}
