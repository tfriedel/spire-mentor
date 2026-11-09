import { describe, expect, it } from "vitest";
import { parseFollowParams, parseOpenParam, resolveRunPath, runLogName, seekIndex } from "./followUrl";

describe("parseFollowParams", () => {
  it("reads follow, run and step", () => {
    expect(parseFollowParams("?follow=1&run=a.jsonl&step=7")).toEqual({
      follow: true,
      run: "a.jsonl",
      step: 7,
    });
  });
  it("leaves a missing step null", () => {
    expect(parseFollowParams("?follow=1")).toEqual({ follow: true, run: null, step: null });
  });
  it("treats a non-numeric, negative or fractional step as null", () => {
    expect(parseFollowParams("?follow=1&step=abc").step).toBeNull();
    expect(parseFollowParams("?follow=1&step=-2").step).toBeNull();
    expect(parseFollowParams("?follow=1&step=2.5").step).toBeNull();
    expect(parseFollowParams("?follow=1&step=").step).toBeNull();
  });
  it("is not a follow link without follow=1", () => {
    expect(parseFollowParams("?run=a.jsonl&step=3").follow).toBe(false);
  });
});

describe("seekIndex", () => {
  it("opens at the newest step without a step", () => {
    expect(seekIndex(null, 5)).toBe(4);
  });
  it("clamps a step past the end", () => {
    expect(seekIndex(9, 5)).toBe(4);
  });
  it("seeks to an in-range step", () => {
    expect(seekIndex(2, 5)).toBe(2);
  });
  it("is 0 for an empty run", () => {
    expect(seekIndex(null, 0)).toBe(0);
    expect(seekIndex(3, 0)).toBe(0);
  });
});

describe("resolveRunPath", () => {
  const live = String.raw`C:\Users\me\sts2-pycli\logs\20261020_013259_Ironclad_NM5B.jsonl`;
  it("names a path the way the control plane does", () => {
    expect(runLogName(live)).toBe("20261020_013259_Ironclad_NM5B.jsonl");
    expect(runLogName("/x/a b.jsonl")).toBe("a_b.jsonl");
  });
  it("opens a link to the live run by its live path", () => {
    expect(resolveRunPath("20261020_013259_Ironclad_NM5B.jsonl", live)).toBe(live);
  });
  it("keeps a link to another run, and falls back to the live run", () => {
    expect(resolveRunPath("other.jsonl", live)).toBe("other.jsonl");
    expect(resolveRunPath(null, live)).toBe(live);
    expect(resolveRunPath(null, null)).toBeNull();
  });
});

describe("parseOpenParam", () => {
  it("reads the run log to open", () => {
    expect(parseOpenParam("?open=a%2Fb_game_1SNV.jsonl")).toBe("a/b_game_1SNV.jsonl");
    expect(parseOpenParam("?open=")).toBeNull();
    expect(parseOpenParam("?open=%20")).toBeNull();
    expect(parseOpenParam("?follow=1")).toBeNull();
  });
});
