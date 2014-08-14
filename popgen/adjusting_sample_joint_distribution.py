import pandas as pd
import numpy as np


def hhld_0_joint_dist(hhid_var, hh_sample, hh_var_list):
    hhld_0_joint_dist = hhid_var
    dist_values = hh_sample
    dist_values['frequency'] = 1
    agg = dict.fromkeys(hh_var_list, 'min')
    agg['frequency'] = 'sum'
    agg['group_id'] = 'min'
    dist_values = dist_values.groupby('hhld_uniqueid', as_index=False).agg(agg)
    hhld_0_joint_dist = pd.merge(
        hhld_0_joint_dist,
        dist_values,
        how='left',
        on='group_id',
        suffixes=(
            '',
            '_y'))
    hhld_0_joint_dist['pumano'] = 0
    hhld_0_joint_dist['county'] = 0
    hhld_0_joint_dist['tract'] = 0
    hhld_0_joint_dist['bg'] = 0
    cols = ['pumano', 'county', 'tract', 'bg']
    cols.extend(hh_var_list)
    cols.extend(['frequency', 'hhld_uniqueid'])
    hhld_0_joint_dist = hhld_0_joint_dist[cols]
    hhld_0_joint_dist.frequency = hhld_0_joint_dist.frequency.fillna(0)
    return hhld_0_joint_dist


def person_0_joint_dist(pid_var, per_sample, per_vars):
    person_0_joint_dist = pid_var
    dist_values = per_sample
    dist_values['frequency'] = 1
    agg = dict.fromkeys(per_vars, 'min')
    agg['frequency'] = 'sum'
    agg['group_id'] = 'min'
    dist_values = dist_values.groupby(
        'person_uniqueid',
        as_index=False).agg(agg)
    person_0_joint_dist = pd.merge(
        person_0_joint_dist,
        dist_values,
        how='left',
        on='group_id',
        suffixes=(
            '',
            '_y'))
    person_0_joint_dist['pumano'] = 0
    person_0_joint_dist['county'] = 0
    person_0_joint_dist['tract'] = 0
    person_0_joint_dist['bg'] = 0
    cols = ['pumano', 'county', 'tract', 'bg']
    cols.extend(per_vars)
    cols.extend(['frequency', 'person_uniqueid'])
    person_0_joint_dist = person_0_joint_dist[cols]
    person_0_joint_dist.frequency = person_0_joint_dist.frequency.fillna(0)
    return person_0_joint_dist


def create_joint_dist_person(pumano, pid_var, per_vars, per_sample):
    jd = per_sample[per_sample.pumano == pumano]
    agg = dict.fromkeys(per_vars, 'min')
    agg['frequency'] = 'sum'
    agg['pumano'] = 'min'
    if len(jd.index) > 0:
        jd['frequency'] = 1
        jd = jd.groupby('person_uniqueid', as_index=False).agg(agg)
        jd = pd.merge(
            pid_var,
            jd,
            how='left',
            on='person_uniqueid',
            suffixes=(
                '',
                '_x'))
        jd.frequency = jd.frequency.fillna(0)
        jd.pumano = jd.pumano.fillna(pumano)
        jd['county'] = 0
        jd['tract'] = 0
        jd['bg'] = 0
        cols = ['pumano', 'county', 'tract', 'bg']
        cols.extend(per_vars)
        cols.extend(['frequency', 'person_uniqueid'])
        jd = jd[cols]
        return jd


def create_joint_dist_hhld(pumano, hhid_var, hh_var_list, hh_sample):
    jd = hh_sample[hh_sample.pumano == pumano]
    agg = dict.fromkeys(hh_var_list, 'min')
    agg['frequency'] = 'sum'
    agg['pumano'] = 'min'
    if len(jd.index) > 0:
        jd['frequency'] = 1
        jd = jd.groupby('hhld_uniqueid', as_index=False).agg(agg)
        jd = pd.merge(
            hhid_var,
            jd,
            how='left',
            on='hhld_uniqueid',
            suffixes=(
                '',
                '_x'))
        jd.frequency = jd.frequency.fillna(0)
        jd.pumano = jd.pumano.fillna(pumano)
        jd['county'] = 0
        jd['tract'] = 0
        jd['bg'] = 0
        cols = ['pumano', 'county', 'tract', 'bg']
        cols.extend(hh_var_list)
        cols.extend(['frequency', 'hhld_uniqueid'])
        jd = jd[cols]
        return jd


def adjust_person_frequencies(jd, pums_jd, state, county, tract, bg):
    dummy = pd.DataFrame(columns=jd.columns)
    jd['puma_prob'] = np.divide(
        jd.frequency.astype('float'),
        jd.frequency.sum())
    jd['upper_prob_bound'] = np.divide(0.5, jd.frequency.sum())
    pums_jd['pums_prob'] = np.divide(
        pums_jd.frequency.astype('float'),
        pums_jd.frequency.sum())
    pums_jd = pums_jd[['person_uniqueid', 'pums_prob']]
    jd = pd.merge(jd, pums_jd, how='left', on='person_uniqueid')
    jd['puma_adjustment'] = jd.pums_prob
    jd['puma_adjustment'][
        jd.pums_prob > jd.upper_prob_bound] = jd['upper_prob_bound'][
        jd.pums_prob > jd.upper_prob_bound]
    correction = 1 - (jd.puma_adjustment[jd.puma_prob == 0].sum())
    jd.puma_prob[
        jd.puma_prob != 0] = np.multiply(
        jd.puma_prob[
            jd.puma_prob != 0],
        correction)
    jd.puma_prob[jd.puma_prob == 0] = jd.puma_adjustment[jd.puma_prob == 0]
    jd['state'] = state
    jd['county'] = county
    jd['tract'] = tract
    jd['bg'] = bg
    freq_sum = int(jd.frequency.sum())
    jd.frequency = np.multiply(jd.puma_prob, freq_sum)
    jd = jd.drop(['upper_prob_bound',
                  'puma_prob',
                  'pums_prob',
                  'puma_adjustment'],
                 axis=1)
    dummy = pd.concat([dummy, jd])
    return dummy


def adjust_hhld_frequencies(jd, pums_jd, state, county, tract, bg):
    dummy = pd.DataFrame(columns=jd.columns)
    jd['puma_prob'] = np.divide(
        jd.frequency.astype('float'),
        jd.frequency.sum())
    jd['upper_prob_bound'] = np.divide(0.5, jd.frequency.sum())
    pums_jd['pums_prob'] = np.divide(
        pums_jd.frequency.astype('float'),
        pums_jd.frequency.sum())
    pums_jd = pums_jd[['hhld_uniqueid', 'pums_prob']]
    jd = pd.merge(jd, pums_jd, how='left', on='hhld_uniqueid')
    jd['puma_adjustment'] = 0
    jd['puma_adjustment'][
        jd.pums_prob <= jd.upper_prob_bound] = jd.pums_prob[
        jd.pums_prob <= jd.upper_prob_bound]
    jd['puma_adjustment'][
        jd.pums_prob > jd.upper_prob_bound] = jd.upper_prob_bound[
        jd.pums_prob > jd.upper_prob_bound]
    correction = 1 - (jd.puma_adjustment[jd.puma_prob == 0].sum())
    jd.puma_prob[
        jd.puma_prob != 0] = np.multiply(
        jd.puma_prob[
            jd.puma_prob != 0],
        correction)
    jd.puma_prob[jd.puma_prob == 0] = jd.puma_adjustment[jd.puma_prob == 0]
    jd['state'] = state
    jd['county'] = county
    jd['tract'] = tract
    jd['bg'] = bg
    jd.frequency = np.multiply(jd.frequency.sum(), jd.puma_prob)
    jd = jd.drop(['upper_prob_bound',
                  'puma_prob',
                  'pums_prob',
                  'puma_adjustment'],
                 axis=1)
    dummy = pd.concat([dummy, jd])
    return dummy


def zero_marginal_adjustment(marginals):
    c = 0
    # identifying number of zeros
    for name in marginals:
        if marginals[name] == 0:
            c += 1

    # if no zero marginals send back original
    if c == 0:
        return marginals

    # zero marginal adjustment
    adj = 0.1 / c

    # replacing zeros with adjustment

    # print 'old marginals', marginals
    for name in marginals:
        if marginals[name] == 0:
            marginals[name] = adj
    # print 'zero marginal adjusted ', marginals
    return marginals


def tolerance(adjustment_all, adjustment_old, iteration, ipf_tolerance):
    adjustment_all = np.array(adjustment_all)
    adjustment_old = np.array(adjustment_old)
    adjustment_difference = abs(adjustment_all - adjustment_old)
    adjustment_convergence_characteristic = adjustment_difference.cumsum()[-1]
    if adjustment_convergence_characteristic > ipf_tolerance:
        return 1
    else:
        return 0


def prepare_control_marginals(
        varcorrdict, marginals, state, county, tract, bg):
    marginals = marginals[
        (marginals.state == state) & (
            marginals.county == county) & (
            marginals.tract == tract) & (
                marginals.bg == bg)]
    control_marginals = {}
    for name in varcorrdict:
        marg_val = int(marginals[name])
        if marg_val > 0:
            control_marginals[name] = marg_val
        else:
            control_marginals[name] = 0
    control_marginals = zero_marginal_adjustment(control_marginals)
    return control_marginals


def marginals_prep(name, joint_dist, varcorrdict):
    var = varcorrdict[name][0]
    cat = varcorrdict[name][1]
    cat_frequency = joint_dist.groupby(var, as_index=False)['frequency'].sum()
    result = float(cat_frequency.frequency[cat_frequency[var] == cat])
    return result


def update_weights(vals, joint_dist):
    col = vals[0]
    val = vals[1]
    adjustment = vals[2]
    joint_dist.frequency[
        joint_dist[col] == val] = np.multiply(
        joint_dist.frequency[
            joint_dist[col] == val],
        adjustment)
    return joint_dist


def adjust_weights(
        varcorrdict,
        joint_dist,
        marginals,
        ipf_tolerance,
        ipf_iterations,
        state,
        county,
        tract,
        bg,
        pumano):
    control_marginals = prepare_control_marginals(
        varcorrdict,
        marginals,
        state,
        county,
        tract,
        bg)
    tol = 1
    iteration = 0
    adjustment_old = []
    target_adjustment = []
    while tol:
        iteration += 1
        adjustment_all = []
        update_list = []
        for name in varcorrdict:
            col_val = varcorrdict[name]
            marginal = marginals_prep(name, joint_dist, varcorrdict)
            adjustment = float(np.divide(control_marginals[name], marginal))
            vals = (col_val[0], col_val[1], adjustment)
            joint_dist = update_weights(vals, joint_dist)
            # update_list.append(vals)
            adjustment_all.append(adjustment)
            if iteration == 1:
                if adjustment == 0:
                    adjustment_old.append(0)
                else:
                    adjustment_old.append(1)
                target_adjustment = [adjustment_old]

        tol = tolerance(
            adjustment_all,
            adjustment_old,
            iteration,
            ipf_tolerance)
        adjustment_old = adjustment_all
        adjustment_characteristic = abs(
            np.array(adjustment_all) - np.array(target_adjustment)).sum() / len(adjustment_all)
    if iteration >= ipf_iterations:
        print "maximum iterations reached"
        pass
    else:
        print "convergence achieved in %s iterations" % (str(iteration))
    return joint_dist


def hhld_estimated_constraint(hhld_joint_dists, geogs):
    hhld_estimated_constraint = {}
    for name in hhld_joint_dists:
        puma_id = int(name[5:8])
        puma_name = 'ec_' + str(puma_id)
        hhld_joint_dist = hhld_joint_dists[name]
        geogs = zip(
            list(geocorr.county[geocorr.pumano == puma_id]),
            list(geocorr.tract[geocorr.pumano == puma_id]),
            list(geocorr.bg[geocorr.pumano == puma_id]))
        hhld_estimated_constraint[puma_name] = {}
        for county, tract, bg in geogs:
            bg_jd = hhld_joint_dist[(hhld_joint_dist.county == county) & (
                hhld_joint_dist.tract == tract) & (hhld_joint_dist.bg == bg)]
            bg_jd = bg_jd.sort(columns=hh_var_list)
            bg_jd = bg_jd[['frequency']]
            ec_id = '%s, %s, %s' % (county, tract, bg)
            hhld_estimated_constraint[puma_name][ec_id] = bg_jd
    return hhld_estimated_constraint


def person_estimated_constraint(person_joint_dists, geogs):
    person_estimated_constraint = {}
    for name in person_joint_dists:
        puma_id = int(name[7:10])
        puma_name = 'ec_' + str(puma_id)
        person_joint_dist = person_joint_dists[name]
        geogs = zip(
            list(geocorr.county[geocorr.pumano == puma_id]),
            list(geocorr.tract[geocorr.pumano == puma_id]),
            list(geocorr.bg[geocorr.pumano == puma_id]))
        person_estimated_constraint[puma_name] = {}
        for county, tract, bg in geogs:
            bg_jd = person_joint_dist[
                (person_joint_dist.county == county) & (
                    person_joint_dist.tract == tract) & (
                    person_joint_dist.bg == bg)]
            bg_jd = bg_jd.sort(columns=per_vars)
            bg_jd = bg_jd[['frequency']]
            ec_id = '%s, %s, %s' % (county, tract, bg)
            person_estimated_constraint[puma_name][ec_id] = bg_jd
    return person_estimated_constraint


def create_joint_dist():
    housing_synthetic_data = pd.DataFrame(
        columns=[
            'state',
            'county',
            'tract',
            'bg',
            'hhid',
            'serialno',
            'frequency',
            'hhuniqueid'])
    person_synthetic_data = pd.DataFrame(
        columns=[
            'state',
            'county',
            'tract',
            'bg',
            'hhid',
            'serialno',
            'pnum',
            'frequency',
            'personuniqueid'])
