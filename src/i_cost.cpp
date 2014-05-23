/*
 * File description: i_cost.cpp
 * Author information: Mike Ranzinger mranzinger@alchemyapi.com
 * Copyright information: Copyright Orchestr8 LLC
 */

#include "i_cost.h"

using namespace std;

const string CostMap::PRIMARY_NAME("cost");

CostMap::CostMap(std::initializer_list<value_type> initList)
    : map<string,Real>(move(initList))
{
}

CostMap& CostMap::operator +=(const CostMap& mp)
{
    for (pair<const string, Real> &kv : *this)
    {
        auto iter = mp.find(kv.first);

        if (iter != mp.end())
            kv.second += iter->second;
    }

    for (const pair<string, Real> &kv : mp)
    {
        auto iter = find(kv.first);

        if (iter == end())
            insert(kv);
    }

    return *this;
}

CostMap& CostMap::operator *=(Real val)
{
    for (pair<const string, Real> &kv : *this)
        kv.second *= val;
    return *this;
}

CostMap& CostMap::operator /=(Real val)
{
    return *this *= 1.0f / val;
}

void BindStruct(const aser::CStructBinder &binder, CostMap &mp)
{
    binder("map", (map<string,Real>&)mp);
}
